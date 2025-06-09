import itertools
import os
from open_spiel.python import policy as policy_module
from open_spiel.python.algorithms import exploitability
import ray
import time
import pyspiel
import collections
import torch
import torch.nn as nn
import numpy as np
from deep_cfr import MLP, ReservoirBuffer, StrategyMemory, AdvantageMemory
from open_spiel.python import policy
from tqdm import tqdm
import wandb
import pickle
@ray.remote
class DeepCFRActor:
    def __init__(self, game, advantage_network, num_traversals_per_actor, memory_capacity, player, batch_size_advantage, batch_size_strategy):
        self.game = game
        self._num_actions = game.num_distinct_actions()
        self.num_traversals_per_actor = num_traversals_per_actor
        self.num_distinct_actions = game.num_distinct_actions()
        self._embedding_size = len(game.new_initial_state().information_state_tensor(0))
        # Create local copies of advantage networks
        self.advantage_network = advantage_network
        # Define advantage network, loss & memory. (One per player)
        self._advantage_memory = ReservoirBuffer(memory_capacity)
        self._strategy_memories = ReservoirBuffer(memory_capacity)
        # Define strategy network, loss & memory.
        self.memory_capacity = memory_capacity
        self.player = player
        self.loss_advantages = nn.MSELoss(reduction="mean")
        self.batch_size_advantage = batch_size_advantage
        self.batch_size_strategy = batch_size_strategy
    def batch_traverse_solve_game(self, iteration):
        """Perform multiple traversals and collect data locally before adding to shared memory."""
        for _ in range(self.num_traversals_per_actor):
            state = self.game.new_initial_state()
            self._traverse_game_tree(state, iteration)
        return self.advantage_network_step()
    
    def policy_network_step(self, policy_network):
        """Begin policy network training."""
        if self.batch_size_strategy:
            strategy_memory_size = len(self._strategy_memories)
            if self.batch_size_strategy > strategy_memory_size:
                return None
            samples = self._strategy_memories.sample(self.batch_size_strategy)
        else:
            memory_size = len(self._strategy_memories)
            if memory_size == 0:
                return None
            samples = self._strategy_memories.sample(memory_size)
        
        if not samples:
            return None
        
        info_states = []
        action_probs = []
        iterations = []
        for s in samples:
            info_states.append(s.info_state)
            action_probs.append(s.strategy_action_probs)
            iterations.append([s.iteration])

        policy_network.zero_grad()
        iters = torch.FloatTensor(np.sqrt(np.array(iterations)))
        ac_probs = torch.FloatTensor(np.array(np.squeeze(action_probs)))
        logits = self._policy_network(torch.FloatTensor(np.array(info_states)))
        outputs = self._policy_sm(logits)
        loss_strategy = self._loss_policy(iters * outputs, iters * ac_probs)
        loss_strategy.backward()
        # here we need to return the gradients to be aggregated at the parameter server
        return [p.grad.clone() for p in policy_network.parameters()], loss_strategy.detach().numpy()
        # here learn the advantage network
    def _traverse_game_tree(self, state, iteration):
      """Performs a traversal of the game tree.

      Over a traversal the advantage and strategy memories are populated with
      computed advantage values and matched regrets respectively.

      Args:
        state: Current OpenSpiel game state.
        player: (int) Player index for this traversal.

      Returns:
        (float) Recursively returns expected payoffs for each action.
      """
      expected_payoff = collections.defaultdict(float)
      if state.is_terminal():
        # Terminal state get returns.
        return state.returns()[self.player]
      elif state.is_chance_node():
        # If this is a chance node, sample an action
        chance_outcome, chance_proba = zip(*state.chance_outcomes())
        action = np.random.choice(chance_outcome, p=chance_proba)
        return self._traverse_game_tree(state.child(action), iteration)
      elif state.current_player() == self.player:
        sampled_regret = collections.defaultdict(float)
        # Update the policy over the info set & actions via regret matching.
        _, strategy = self._sample_action_from_advantage(state, self.player)
        for action in state.legal_actions():
          expected_payoff[action] = self._traverse_game_tree(
              state.child(action), iteration)
        cfv = 0
        for a_ in state.legal_actions():
          cfv += strategy[a_] * expected_payoff[a_]
        for action in state.legal_actions():
          sampled_regret[action] = expected_payoff[action]
          sampled_regret[action] -= cfv
        sampled_regret_arr = [0] * self._num_actions
        for action in sampled_regret:
          sampled_regret_arr[action] = sampled_regret[action]
        self._advantage_memory.add(
            AdvantageMemory(np.array(state.information_state_tensor()), np.array(iteration),
                            np.array(sampled_regret_arr)))
        return cfv
      else:
        other_player = state.current_player()
        _, strategy = self._sample_action_from_advantage(state, other_player)
        # Recompute distribution for numerical errors.
        probs = np.array(strategy)
        probs /= probs.sum()
        sampled_action = np.random.choice(range(self._num_actions), p=probs)
        self._strategy_memories.add(
          StrategyMemory(
              np.array(state.information_state_tensor(other_player)), np.array(iteration),
              np.array(strategy)))
        return self._traverse_game_tree(state.child(sampled_action), iteration)

    def _sample_action_from_advantage(self, state, player):
        """Sample action from advantage using local network copy."""
        info_state = state.information_state_tensor(player)
        legal_actions = state.legal_actions(player)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(np.expand_dims(info_state, axis=0))
            raw_advantages = self.advantage_network(state_tensor)[0].numpy()
        
        advantages = np.maximum(0., raw_advantages)
        cumulative_regret = np.sum(advantages[legal_actions])
        
        matched_regrets = np.zeros(self.num_distinct_actions)
        if cumulative_regret > 0.:
            for action in legal_actions:
                matched_regrets[action] = advantages[action] / cumulative_regret
        else:
            best_action = max(legal_actions, key=lambda a: raw_advantages[a])
            matched_regrets[best_action] = 1.0
            
        return advantages, matched_regrets
    
    def advantage_network_step(self):
        """Optimized advantage network training with pre-allocated arrays."""
        # a = time.time()
        if self.batch_size_advantage:
            memory_size = len(self._advantage_memory)
            if self.batch_size_advantage > memory_size:
                return None
            samples = self._advantage_memory.sample(self.batch_size_advantage)
        else:
            memory_size = len(self._advantage_memory)
            if memory_size == 0:
                return None
            samples = self._advantage_memory.sample(memory_size)
        
        if not samples:
            return None
        # print("time to sample: ", time.time()-a)
        # Pre-allocate numpy arrays for better performance
        # a = time.time()
        batch_size = len(samples)
        info_state_size = len(samples[0].info_state)
        advantage_size = len(samples[0].advantage)
        
        info_states = np.empty((batch_size, info_state_size), dtype=np.float32)
        advantages = np.empty((batch_size, advantage_size), dtype=np.float32)
        iterations = np.empty((batch_size, 1), dtype=np.float32)
        
        # Vectorized data extraction
        for i, s in enumerate(samples):
            info_states[i] = s.info_state
            advantages[i] = s.advantage
            iterations[i] = s.iteration
        self.advantage_network.zero_grad()
        advantages_tensor = torch.from_numpy(advantages)
        iters_tensor = torch.from_numpy(np.sqrt(iterations))
        states_tensor = torch.from_numpy(info_states)
        # print("preparing data: ", time.time()-a)
        # a = time.time()
        outputs = self.advantage_network(states_tensor)
        loss_advantages = self.loss_advantages(iters_tensor * outputs,
                                                iters_tensor * advantages_tensor)
        loss_advantages.backward()
        # get the gradients to be aggregated at the parameter server
        return [p.grad.clone() for p in self.advantage_network.parameters()], self.player, loss_advantages.detach().numpy()

class Orchestrator:
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
        self.parameter_servers = [ParameterServer(game, policy_network_layers, advantage_network_layers, learning_rate, player) for player in range(self.game.num_players())]
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
    def _initialize_actors(self, num_actors):
        """Initialize actors with current network parameters"""
        # Calculate traversals per actor
        self.num_traversals_per_actor = max(1, self.num_traversals // self.num_actors) * 2
        self.actors = []
        # create a list of half for player 1 and the other half for player 2
        for player in range(self.game.num_players()):
            self.actors += [DeepCFRActor.remote(game, self.parameter_servers[player]._advantage_network_ref, self.num_traversals_per_actor, self.memory_capacity//(num_actors//2), player, self.batch_size_advantage//(num_actors//2)) for _ in range(num_actors//self.game.num_players())]

    def solve(self, use_wandb = False):
        """Solution logic for Deep CFR."""
        advantage_losses = collections.defaultdict(list)
        # wandb_config = {"lr": self.learning_rate, "batch_size": self._batch_size_advantage, "memory_capacity": self.memory_capacity, "policy_network_train_steps": self._policy_network_train_steps, "advantage_network_train_steps": self._advantage_network_train_steps, "num_actors": self.num_actors, "num_traversals": self._num_traversals, "num_iterations": self._num_iterations, "evaluation_interval": self.evaluation_interval, "num_players": self._num_players, "policy_network_layers": self.policy_network_layers, "advantage_network_layers": self.advantage_network_layers, "reinitialize_advantage_networks": self._reinitialize_advantage_networks}   
        self._initialize_actors(self.num_actors - 2)
        
        if use_wandb:
            run = wandb.init(project="deep_cfr_ray", config=wandb_config)
        running_return = 0
        for i in tqdm(range(self.num_iterations), desc="CFR Iterations"):
            # Initialize actors with current network state
            # initialize parameter servers
            # Parallel traversals
            traversal_tasks = [actor.batch_traverse_solve_game.remote(
                i) 
                for actor in self.actors]
            aggregated_losses = [[], []]
            # Wait for all traversals to complete and collect results
            while traversal_tasks:
                # Wait for at least one task to complete
                done_ids, traversal_tasks = ray.wait(traversal_tasks)
                # Get the results from completed tasks
                output = ray.get(done_ids)
                # Get actual data from references
                # Add data to memories
                for gradient, player, loss in output:
                    aggregated_losses[player].append(loss)
                    self.parameter_servers[player].add_gradients(gradient)
            # Reinitialize advantage networks
            if self.reinitialize_advantage_networks:
                for player in range(self.game.num_players()):
                    self.parameter_servers[player].reinitialize_advantage_network()
            # Train advantage network
            # aggregate gradients
            for parameter_server in self.parameter_servers:
                parameter_server.aggregate_gradients()
            # breakpoint()
            for player in range(self.game.num_players()):
                advantage_losses[player].append(np.mean(aggregated_losses[player]))
                self.parameter_servers[player]._advantage_network_ref = ray.put(self.parameter_servers[player]._advantage_network)
            if i % self.evaluation_interval == 0:
                for _ in range(self.policy_network_train_steps):
                    gradients, loss = [actor.policy_network_step.remote(self._policy_network) for actor in self.actors]
                    gradients = ray.get(gradients)
                    loss = ray.get(loss)
                    self._policy_network.zero_grad()
                    for p, g in zip(self._policy_network.parameters(), gradients):
                        p.grad = g
                    self._optimizer_policy.step()
            print(f"Advantage loss for player {player}: {advantage_losses[player][-1]}")

        return self._policy_network, advantage_losses, policy_loss
            
    def evaluate_agent(self, num_episodes=100):
        """evaluate the agent on the game against a random agent"""
        self.policy_network_step()
        player_0_returns = np.array([])
        player_1_returns = np.array([])
        for _ in tqdm(range(num_episodes), desc="Evaluating agent"):
            state = self._game.new_initial_state()
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

class ParameterServer(policy.Policy):
    def __init__(self,game, policy_network_layers=(256, 256),
                 advantage_network_layers=(128, 128),
                 learning_rate: float = 1e-4,
                player: int = 0
                ):
        self.game = game
        self._root_node = self.game.new_initial_state()

        self._embedding_size = len(self._root_node.information_state_tensor(0))
        self._num_actions = self.game.num_distinct_actions()
        self._learning_rate = learning_rate
        self.player = player
        # Store advantage network layer configuration to pass to actors
        self._advantage_network_layers_list = list(advantage_network_layers)


        self._advantage_network = MLP(self._embedding_size, self._advantage_network_layers_list, # Use the stored list
                self._num_actions)
        self._advantage_network_ref = ray.put(self._advantage_network)
        self._optimizer_advantage = torch.optim.Adam(
                    self._advantage_network.parameters(), lr=learning_rate)
        self.gradient_buffer = []
    def clear_advantage_buffers(self):
        self._advantage_memories.clear()
    def add_gradients(self, gradients):
        self.gradient_buffer.append(gradients)
    def aggregate_gradients(self):
        grouped_by_param = zip(*self.gradient_buffer)
        
        averaged_gradients = []
        for group in grouped_by_param:
            averaged_gradients.append(torch.stack(group).mean(dim=0))
        # breakpoint()
        for p, g in zip(self._advantage_network.parameters(), averaged_gradients):
            p.grad = g
        self._optimizer_advantage.step()
        self._optimizer_advantage.zero_grad()
        self.gradient_buffer = []
    def reinitialize_advantage_network(self):
        """Reinitialize advantage network for a specific player"""
        self._advantage_network.reset()
        self._optimizer_advantage = torch.optim.Adam(
            self._advantage_network.parameters(), lr=self._learning_rate)
    
    def save_memories(self):
        #mkdir if not there 
        os.makedirs(f"./memories", exist_ok=True)
        # save the advantage memories and strategy memories to a file
        with open(f"./memories/advantage_memories.pkl", "wb") as f:
            pickle.dump(self._advantage_memories, f, pickle.HIGHEST_PROTOCOL)
        with open(f"./memories/strategy_memories.pkl", "wb") as f:
            pickle.dump(self._strategy_memories, f, pickle.HIGHEST_PROTOCOL)
            
    def load_memories(self):
        # load the advantage memories and strategy memories from a file using ray
        with open(f"./memories/advantage_memories.pkl", "rb") as f:
            self._advantage_memories = pickle.load(f)
        with open(f"./memories/strategy_memories.pkl", "rb") as f:
            self._strategy_memories = pickle.load(f)

if __name__ == "__main__":
    try:
        ray.shutdown()
    except:
        pass
    
    ray.init(
        runtime_env={"env_vars": {"RAY_DEBUG": "1"}},
        ignore_reinit_error=True,
    )
    game = pyspiel.load_game('kuhn_poker')
    # Get number of available CPUs for Ray actors
    num_cpus = ray.cluster_resources()['CPU']
    # Leave 1 CPU for the main process
    num_actors = max(1, int(num_cpus) - 1)

    solver = Orchestrator(
    game,
    policy_network_layers=(64,64,64),
    advantage_network_layers=(64,64,64),
    num_iterations=1,
    num_traversals=5000,
    reinitialize_advantage_networks=True,
    learning_rate=1e-3,
    batch_size_advantage=2048,
    batch_size_strategy=2048,
    memory_capacity=1e6,
    policy_network_train_steps=5000,
    advantage_network_train_steps=750,
    evaluation_interval=5,
    num_actors=num_actors
    )
    _, advantage_losses, policy_loss = solver.solve(use_wandb=False)
    
    for player, losses in list(advantage_losses.items()):
        print("Advantage for player:", player,
                      losses[:2] + ["..."] + losses[-2:])
        print("Advantage Buffer Size for player", player,
                      len(solver.advantage_buffers[player]))
    print("Strategy Buffer Size:",
                    len(solver.strategy_buffer))
    print("Final policy loss:", policy_loss)
    def print_policy(policy):
      for state, probs in zip(itertools.chain(*policy.states_per_player),
                              policy.action_probability_array):
        print(f'{state:6}   p={probs}')


    # Get Deep CFR policy
    policy = policy_module.tabular_policy_from_callable(game, solver.action_probabilities)
    conv = exploitability.nash_conv(game, policy)
    print("Deep CFR - NashConv:", conv)  
    np.set_printoptions(precision=3, suppress=True, floatmode='fixed')
    print_policy(policy)
