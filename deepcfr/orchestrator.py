from open_spiel.python import policy as policy_module
from open_spiel.python.algorithms import exploitability
import ray
import collections
import torch
import torch.nn as nn
import numpy as np
from deep_cfr import MLP
from tqdm import tqdm
import wandb
from open_spiel.python import policy
from actor import DeepCFRActor
from parameter_server import ParameterServer

class Orchestrator(policy.Policy):
    def __init__(self, game, policy_network_layers=(256, 256), 
                 advantage_network_layers=(128, 128),
                 num_iterations: int = 100,
                 num_traversals: int = 20,
                 learning_rate: float = 1e-4,
                 batch_size_advantage=None,
                 batch_size_strategy=None,
                 memory_capacity: int = int(1e6),
                 policy_network_train_steps: int = 1,
                 advantage_network_train_steps: int = 1,
                 reinitialize_advantage_networks: bool = True, 
                 evaluation_interval: int = 10,
                 num_actors: int = 0,
                 ):
        self.game = game
        self.policy_network_layers = policy_network_layers
        self.advantage_network_layers = advantage_network_layers
        self.num_iterations = num_iterations
        self.num_traversals = num_traversals
        self.learning_rate = learning_rate
        self.batch_size_advantage = batch_size_advantage
        self.batch_size_strategy = batch_size_strategy
        self.num_actors = num_actors
        # create the parameter servers for each player 
        self.policy_network_train_steps = policy_network_train_steps
        self.parameter_servers = [ParameterServer.options(name=f"parameter_server_{player}", lifetime="detached").remote(game, policy_network_layers, advantage_network_layers, learning_rate, player) for player in range(self.game.num_players())]
        self.memory_capacity = memory_capacity
        self.reinitialize_advantage_networks = reinitialize_advantage_networks
        self.evaluation_interval = evaluation_interval
        self._embedding_size = len(self.game.new_initial_state().information_state_tensor(0))
        self._num_actions = self.game.num_distinct_actions()
        self._policy_network = MLP(self._embedding_size,
                                  list(policy_network_layers),
                                  self._num_actions)
        self._policy_sm = nn.Softmax(dim=-1)
        self._loss_policy = nn.MSELoss()
        self._optimizer_policy = torch.optim.Adam(
            self._policy_network.parameters(), lr=learning_rate)
        self._policy_network_ref = ray.put(self._policy_network)
        self.advantage_network_train_steps = advantage_network_train_steps
    def _initialize_actors(self, num_actors):
        """Initialize actors with current network parameters"""
        # Calculate traversals per actor
        self.num_traversals_per_actor = max(1, self.num_traversals // (self.num_actors-2))
        self.actors = []
        # create a list of half for player 1 and the other half for player 2
        for player in range(self.game.num_players()):
            self.actors += [DeepCFRActor.options(name=f"actor_{player}_{i}", lifetime="detached", namespace="deep_cfr").remote(self.game, self.num_traversals_per_actor, self.memory_capacity, self.batch_size_advantage, self.batch_size_strategy, self._policy_sm, self._loss_policy, player) for i in range(num_actors//self.game.num_players())]

    def solve(self, use_wandb = False):
        """Solution logic for Deep CFR."""
        advantage_losses = collections.defaultdict(list)
        # wandb_config = {"lr": self.learning_rate, "batch_size": self._batch_size_advantage, "memory_capacity": self.memory_capacity, "policy_network_train_steps": self._policy_network_train_steps, "advantage_network_train_steps": self._advantage_network_train_steps, "num_actors": self.num_actors, "num_traversals": self._num_traversals, "num_iterations": self._num_iterations, "evaluation_interval": self.evaluation_interval, "num_players": self._num_players, "policy_network_layers": self.policy_network_layers, "advantage_network_layers": self.advantage_network_layers, "reinitialize_advantage_networks": self._reinitialize_advantage_networks}   
        
        self._initialize_actors(self.num_actors - 2)
        if use_wandb:
            run = wandb.init(project="deep_cfr_ray", config=wandb_config)
        running_return = 0
        for i in tqdm(range(self.num_iterations), desc="CFR Iterations"):
            # Parallel traversals
            traversal_tasks = []
            for player in range(self.game.num_players()):
                actors = [ray.get_actor(f"actor_{player}_{i}", namespace="deep_cfr")
                          for i in range((self.num_actors - 2) // self.game.num_players())]

                # Gather fresh copies of ALL players' advantage networks once per
                # traversal iteration and broadcast them to the actors.  Each
                # actor needs access to every player's network in order to
                # sample opponent actions correctly.
                all_networks = [ray.get(self.parameter_servers[p].put_network.remote())
                                for p in range(self.game.num_players())]

                traversal_tasks += [actor.batch_traverse_solve_game.remote(i, all_networks)
                                     for actor in actors]
            
            # Reinitialize advantage networks
            if self.reinitialize_advantage_networks:
                for player in range(self.game.num_players()):
                    self.parameter_servers[player].reinitialize_advantage_network.remote()
            aggregated_losses = [[] for _ in range(self.game.num_players())]
            # Wait for all traversals to complete and collect results
            output = ray.get(traversal_tasks)
            for _ in range(self.advantage_network_train_steps):
                outputs = []
                for player in range(self.game.num_players()):
                    actors = [ray.get_actor(f"actor_{player}_{i}", namespace="deep_cfr")
                              for i in range((self.num_actors - 2) // self.game.num_players())]
                    outputs += [actor.advantage_network_step.remote(ray.get(self.parameter_servers[player].put_network.remote()))
                                for actor in actors]
                raw_results = ray.get(outputs)
                # Filter out actors that did not return a gradient (e.g. due to
                # insufficient samples).
                valid_results = [r for r in raw_results if r is not None]
                if not valid_results:
                    # Skip update if nobody had enough data yet.
                    continue

                gradient_buffer = {k: [] for k in range(self.game.num_players())}
                for gradient, plyr, loss in valid_results:
                    aggregated_losses[plyr].append(loss)
                    gradient_buffer[plyr].append(gradient)
                # aggregate gradients
                # Train advantage network
                for plyr in range(self.game.num_players()):
                    if gradient_buffer[plyr]:
                        self.parameter_servers[plyr].aggregate_gradients.remote(gradient_buffer[plyr])

            for player in range(self.game.num_players()):
                advantage_losses[player].append(np.mean(aggregated_losses[player]))
                self.parameter_servers[player].put_network.remote()
            
            for player in range(self.game.num_players()):
                print(f"Advantage loss for player {player}: {advantage_losses[player][-1]}")
                
            if i % self.evaluation_interval == 0:
                policy_losses = self.train_policy_network()
                print(f"Policy loss: {policy_losses[-1]}")
                player_0_returns, player_1_returns = self.evaluate_agent()
                print(f"Player 0 returns: {player_0_returns}, Player 1 returns: {player_1_returns}")
                policy = policy_module.tabular_policy_from_callable(self.game, self.action_probabilities)
                conv = exploitability.nash_conv(self.game, policy)
                self._policy_network.reset()
                self._optimizer_policy = torch.optim.Adam(self._policy_network.parameters(), lr=self.learning_rate)
                print("Deep CFR - NashConv:", conv)  

        return self._policy_network, advantage_losses, policy_losses
    
    def train_policy_network(self):
        """Train the policy network via gradient aggregation across all actors."""
        policy_losses = []

        for _ in range(self.policy_network_train_steps):
            # Collect gradients and losses from ALL actors of BOTH players.
            remote_tasks = []
            for player in range(self.game.num_players()):
                actors = [ray.get_actor(f"actor_{player}_{i}", namespace="deep_cfr")
                          for i in range((self.num_actors - 2) // self.game.num_players())]
                remote_tasks += [actor.policy_network_step.remote(self._policy_network) for actor in actors]

            # Fetch results. Some actors might return `None` if their strategy
            # buffer is still too small – filter those out.
            results = [res for res in ray.get(remote_tasks) if res is not None]
            if not results:
                # Not enough data yet – skip this optimisation step.
                continue

            gradients_list, losses_list = zip(*results)
            policy_losses.append(float(np.mean(losses_list)))

            # Average gradients across all actors.
            averaged_grads = []
            for grads_per_param in zip(*gradients_list):
                averaged_grads.append(torch.stack(grads_per_param).mean(dim=0))

            # Apply gradients to the local policy network.
            self._optimizer_policy.zero_grad()
            for param, grad in zip(self._policy_network.parameters(), averaged_grads):
                param.grad = grad
            self._optimizer_policy.step()

        return policy_losses
                
    def evaluate_agent(self, num_episodes=100):
        """evaluate the agent on the game against a random agent"""
        player_0_returns = np.array([])
        player_1_returns = np.array([])
        for _ in tqdm(range(num_episodes), desc="Evaluating agent"):
            state = self.game.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    chance_outcome, chance_proba = zip(*state.chance_outcomes())
                    action = np.random.choice(chance_outcome, p=chance_proba)
                elif state.current_player() == 0:
                    action = self.action_probabilities(state)
                    # renormalize 
                    action = {k: v / sum(action.values()) for k, v in action.items()}
                    action = np.random.choice(list(action.keys()), p=list(action.values()))
                else:
                    # take random action
                    action = np.random.choice(state.legal_actions())
                state = state.child(action)
            player_0_returns = np.append(player_0_returns, state.returns()[0])
            player_1_returns = np.append(player_1_returns, state.returns()[1])
        return np.sum(player_0_returns) / num_episodes, np.sum(player_1_returns) / num_episodes

    @property
    def advantage_buffers(self):
        return self._advantage_memories

    @property
    def strategy_buffer(self):
        return self._strategy_memories
    
    def calculate_exploitability(self):
        """Compute exploitability of the policy."""
        policy = policy_module.tabular_policy_from_callable(self.game, self.action_probabilities)
        return exploitability.nash_conv(self.game, policy)
    
   
    def action_probabilities(self, state):
        """Computes action probabilities for the current player in state."""
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        info_state_vector = np.array(state.information_state_tensor())
        if len(info_state_vector.shape) == 1:
            info_state_vector = np.expand_dims(info_state_vector, axis=0)
        with torch.no_grad():
            logits = self._policy_network(torch.FloatTensor(info_state_vector))
            probs = self._policy_sm(logits).numpy()
        return {action: probs[0][action] for action in legal_actions}
