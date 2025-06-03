import itertools
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
@ray.remote
class DeepCFRActor:
    def __init__(self, game, advantage_networks_state_dicts, num_traversals_per_actor, advantage_network_hidden_layers):
        self.game = game
        self._num_actions = game.num_distinct_actions()
        
        self.num_traversals_per_actor = num_traversals_per_actor
        self.num_distinct_actions = game.num_distinct_actions()
        self._embedding_size = len(game.new_initial_state().information_state_tensor(0))
        self.advantage_data = []
        self.strategy_data = []
        # Create local copies of advantage networks
        self.advantage_networks = []
        for i, state_dict in enumerate(advantage_networks_state_dicts):
            net = MLP(self._embedding_size, advantage_network_hidden_layers, self.num_distinct_actions)  # Use passed hidden layers
            net.load_state_dict(state_dict)
            net.eval()  # Set to evaluation mode
            self.advantage_networks.append(net)
        
    def update_advantage_networks(self, advantage_networks_state_dicts):
        """Update local network copies with new parameters"""
        for i, state_dict in enumerate(advantage_networks_state_dicts):
            self.advantage_networks[i].load_state_dict(state_dict)
            self.advantage_networks[i].eval()
        
    def batch_traverse_tree_tasks(self, player, iteration):
        """Perform multiple traversals and collect data locally before adding to shared memory."""
        
        for _ in range(self.num_traversals_per_actor):
            state = self.game.new_initial_state()
            self._traverse_game_tree(state, player, iteration)
        advantage_data_ref = ray.put(self.advantage_data)
        strategy_data_ref = ray.put(self.strategy_data)
        return advantage_data_ref, strategy_data_ref

    def _traverse_game_tree(self, state, player, iteration):
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
        return state.returns()[player]
      elif state.is_chance_node():
        # If this is a chance node, sample an action
        chance_outcome, chance_proba = zip(*state.chance_outcomes())
        action = np.random.choice(chance_outcome, p=chance_proba)
        return self._traverse_game_tree(state.child(action), player, iteration)
      elif state.current_player() == player:
        sampled_regret = collections.defaultdict(float)
        # Update the policy over the info set & actions via regret matching.
        _, strategy = self._sample_action_from_advantage(state, player)
        for action in state.legal_actions():
          expected_payoff[action] = self._traverse_game_tree(
              state.child(action), player, iteration)
        cfv = 0
        for a_ in state.legal_actions():
          cfv += strategy[a_] * expected_payoff[a_]
        for action in state.legal_actions():
          sampled_regret[action] = expected_payoff[action]
          sampled_regret[action] -= cfv
        sampled_regret_arr = [0] * self._num_actions
        for action in sampled_regret:
          sampled_regret_arr[action] = sampled_regret[action]
        self.advantage_data.append(
            AdvantageMemory(state.information_state_tensor(), iteration,
                            sampled_regret_arr))
        return cfv
      else:
        other_player = state.current_player()
        _, strategy = self._sample_action_from_advantage(state, other_player)
        # Recompute distribution for numerical errors.
        probs = np.array(strategy)
        probs /= probs.sum()
        sampled_action = np.random.choice(range(self._num_actions), p=probs)
        self.strategy_data.append(
          StrategyMemory(
              state.information_state_tensor(other_player), iteration,
              strategy))
        return self._traverse_game_tree(state.child(sampled_action), player, iteration)

    def _sample_action_from_advantage(self, state, player):
        """Sample action from advantage using local network copy."""
        info_state = state.information_state_tensor(player)
        legal_actions = state.legal_actions(player)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(np.expand_dims(info_state, axis=0))
            raw_advantages = self.advantage_networks[player](state_tensor)[0].numpy()
        
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
                ):
        try:
            ray.shutdown()
        except:
            pass
        
        ray.init(
            runtime_env={"env_vars": {"RAY_DEBUG": "1"}},
            ignore_reinit_error=True,
        )
        
        self.game = game
        all_players = list(range(self.game.num_players()))
        super(Orchestrator, self).__init__(self.game, all_players)
        self._game = self.game
        
        if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
            raise ValueError("Simulatenous games are not supported.")
            
        self._batch_size_advantage = batch_size_advantage
        self._batch_size_strategy = batch_size_strategy
        self._policy_network_train_steps = policy_network_train_steps
        self._advantage_network_train_steps = advantage_network_train_steps
        self._num_players = self.game.num_players()
        self._root_node = self._game.new_initial_state()
        self._embedding_size = len(self._root_node.information_state_tensor(0))
        self._num_iterations = num_iterations
        self._num_traversals = num_traversals
        self._reinitialize_advantage_networks = reinitialize_advantage_networks
        self._num_actions = self.game.num_distinct_actions()
        self._learning_rate = learning_rate

        # Store advantage network layer configuration to pass to actors
        self._advantage_network_layers_list = list(advantage_network_layers)

        # Define strategy network, loss & memory.
        self._strategy_memories = ReservoirBuffer(memory_capacity)
        self._policy_network = MLP(self._embedding_size,
                                  list(policy_network_layers),
                                  self._num_actions)
        self._policy_sm = nn.Softmax(dim=-1)
        self._loss_policy = nn.MSELoss()
        self._optimizer_policy = torch.optim.Adam(
            self._policy_network.parameters(), lr=learning_rate)

        # Define advantage network, loss & memory. (One per player)
        self._advantage_memories = [
              ReservoirBuffer(memory_capacity) for _ in range(self._num_players)
        ]
        self._advantage_networks = [
            MLP(self._embedding_size, self._advantage_network_layers_list, # Use the stored list
                self._num_actions) for _ in range(self._num_players)
        ]
        self._loss_advantages = nn.MSELoss(reduction="mean")
        self._optimizer_advantages = []
        for p in range(self._num_players):
            self._optimizer_advantages.append(
                torch.optim.Adam(
                    self._advantage_networks[p].parameters(), lr=learning_rate))

    def _initialize_actors(self):
        """Initialize actors with current network parameters"""
        # Get number of available CPUs for Ray actors
        num_cpus = ray.cluster_resources()['CPU']
        # Leave 1 CPU for the main process
        num_actors = max(1, int(num_cpus) - 1)
        self.num_actors = num_actors

        # Calculate traversals per actor
        self.num_traversals_per_actor = max(1, self._num_traversals // self.num_actors)
        
        advantage_networks_state_dicts = [net.state_dict() for net in self._advantage_networks]
        self.actors = [DeepCFRActor.remote(
            self.game, 
            advantage_networks_state_dicts, 
            self.num_traversals_per_actor,
            self._advantage_network_layers_list) # Pass the stored list
                       for _ in range(self.num_actors)]

    def _update_actor_networks(self):
        """Update actor networks with current parameters"""
        advantage_networks_state_dicts = [net.state_dict() for net in self._advantage_networks]
        update_tasks = [actor.update_advantage_networks.remote(advantage_networks_state_dicts) 
                       for actor in self.actors]
        ray.get(update_tasks)

    def clear_advantage_buffers(self):
        for p in range(self._num_players):
            self._advantage_memories[p].clear()
            
    def kill_actors(self):
        for actor in self.actors:
            ray.kill(actor)
    
    def solve(self):
        """Solution logic for Deep CFR."""
        advantage_losses = collections.defaultdict(list)
        
        for i in tqdm(range(self._num_iterations), desc="CFR Iterations"):
            for p in tqdm(range(self._num_players), desc=f"Iteration {i} Players"):
                # Initialize actors with current network state
                a = time.time()
                self._initialize_actors()
                b = time.time()
                print(f"Time taken for initialize actors: {b - a:.2f} seconds")
                a = time.time()
                # Update actor networks with current parameters before traversals
                self._update_actor_networks()
                # Parallel traversals
                b = time.time()
                print(f"Time taken for update actor networks: {b - a:.2f} seconds")
                a = time.time()
                traversal_tasks = [actor.batch_traverse_tree_tasks.remote(
                    p, i) 
                    for actor in self.actors]
                # Wait for all traversals to complete and collect results
                while traversal_tasks:
                    # Wait for at least one task to complete
                    done_ids, traversal_tasks = ray.wait(traversal_tasks)
                    
                    # Get the results from completed tasks
                    for done_id in done_ids:
                        advantage_data_ref, strategy_data_ref = ray.get(done_id)
                        # Get actual data from references
                        advantage_data = ray.get(advantage_data_ref)
                        strategy_data = ray.get(strategy_data_ref)
                        
                        # Add data to memories
                        for data in advantage_data:
                            self._advantage_memories[p].add(data)
                        for data in strategy_data:
                            self._strategy_memories.add(data)
                b = time.time()
                print(f"Time taken for traversals: {b - a:.2f} seconds")

                # Reinitialize advantage networks
                if self._reinitialize_advantage_networks:
                    self.reinitialize_advantage_network(p)
                self.kill_actors()
                a = time.time()
                # Train advantage network
                advantage_losses[p].append(self._learn_advantage_network(p))
                b = time.time()
                print(f"Time taken for learn: {b - a:.2f} seconds")

                print(f"Advantage loss for player {p}: {advantage_losses[p][-1]}")
            
        policy_loss = self._learn_strategy_network()
        
        
        return self._policy_network, advantage_losses, policy_loss

    def reinitialize_advantage_network(self, player):
        """Reinitialize advantage network for a specific player"""
        self._advantage_networks[player].reset()
        self._optimizer_advantages[player] = torch.optim.Adam(
            self._advantage_networks[player].parameters(), lr=self._learning_rate)

    def _learn_strategy_network(self):
        """Compute the loss over the strategy network."""
        for step in range(self._policy_network_train_steps):
            if self._batch_size_strategy:
                strategy_memory_size = len(self._strategy_memories)
                if self._batch_size_strategy > strategy_memory_size:
                    return None
                samples = self._strategy_memories.sample(self._batch_size_strategy)
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

            self._optimizer_policy.zero_grad()
            iters = torch.FloatTensor(np.sqrt(np.array(iterations)))
            ac_probs = torch.FloatTensor(np.array(np.squeeze(action_probs)))
            logits = self._policy_network(torch.FloatTensor(np.array(info_states)))
            outputs = self._policy_sm(logits)
            loss_strategy = self._loss_policy(iters * outputs, iters * ac_probs)
            loss_strategy.backward()
            self._optimizer_policy.step()

        return loss_strategy.detach().numpy()

    @property
    def advantage_buffers(self):
        return self._advantage_memories

    @property
    def strategy_buffer(self):
        return self._strategy_memories

    def _learn_advantage_network(self, player):
        """Optimized advantage network training with pre-allocated arrays."""
        for step in tqdm(range(self._advantage_network_train_steps), desc=f"Training advantage network for player {player}"):
            if self._batch_size_advantage:
                memory_size = len(self._advantage_memories[player])
                if self._batch_size_advantage > memory_size:
                    return None
                samples = self._advantage_memories[player].sample(self._batch_size_advantage)
            else:
                memory_size = len(self._advantage_memories[player])
                if memory_size == 0:
                    return None
                samples = self._advantage_memories[player].sample(memory_size)
          
            if not samples:
                return None
            
            # Pre-allocate numpy arrays for better performance
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
            
            # Convert to tensors efficiently
            self._optimizer_advantages[player].zero_grad()
            advantages_tensor = torch.from_numpy(advantages)
            iters_tensor = torch.from_numpy(np.sqrt(iterations))
            states_tensor = torch.from_numpy(info_states)
            
            outputs = self._advantage_networks[player](states_tensor)
            loss_advantages = self._loss_advantages(iters_tensor * outputs,
                                                  iters_tensor * advantages_tensor)
            loss_advantages.backward()
            self._optimizer_advantages[player].step()

        return loss_advantages.detach().numpy()
    
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

if __name__ == "__main__":
    game = pyspiel.load_game('kuhn_poker')
    solver = Orchestrator(
        game,
        policy_network_layers=(64,64,64),
        advantage_network_layers=(64,64,64),
        num_iterations=101,
        reinitialize_advantage_networks=True,
        num_traversals=1500,
        learning_rate=1e-3,
        batch_size_advantage=256,
        batch_size_strategy=256,
        memory_capacity=1e6,
        policy_network_train_steps=5000,
        advantage_network_train_steps=750,
    )
    import time
    start_time = time.time()

    _, advantage_losses, policy_loss = solver.solve()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
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
