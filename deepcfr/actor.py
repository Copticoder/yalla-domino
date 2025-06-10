import ray
import collections
import torch
import torch.nn as nn
import numpy as np
from deep_cfr import ReservoirBuffer, StrategyMemory, AdvantageMemory

@ray.remote
class DeepCFRActor:
    def __init__(self, game, num_traversals_per_actor, memory_capacity, batch_size_advantage, batch_size_strategy, policy_sm, loss_policy, player):
        self.game = game
        self._num_actions = game.num_distinct_actions()
        self.num_traversals_per_actor = num_traversals_per_actor
        self.num_distinct_actions = game.num_distinct_actions()
        self._embedding_size = len(game.new_initial_state().information_state_tensor(0))
        # Define advantage network, loss & memory. (One per player)
        self._advantage_memory = ReservoirBuffer(memory_capacity)
        self._strategy_memories = ReservoirBuffer(memory_capacity)
        # Define strategy network, loss & memory.
        self.memory_capacity = memory_capacity
        self.loss_advantages = nn.MSELoss(reduction="mean")
        self.batch_size_advantage = batch_size_advantage
        self.batch_size_strategy = batch_size_strategy
        self.policy_sm = policy_sm
        self.loss_policy = loss_policy
        self.player = player
    def batch_traverse_solve_game(self, iteration, advantage_networks):
        """Perform multiple traversals and collect data locally before adding to shared memory.

        Args:
            iteration (int): Current training iteration.
            advantage_networks (List[nn.Module]): A list containing one advantage
                network per player. The actor will look up the correct network
                based on the player index encountered during traversal.
        """
        # Store local copies of the advantage networks for quick access. If the
        # Orchestrator accidentally passes `ObjectRef`s, dereference them
        # defensively here to avoid runtime TypeErrors.
        import ray  # local import to avoid circularities
        self.advantage_networks = [ray.get(net) if isinstance(net, ray.ObjectRef) else net
                                   for net in advantage_networks]

        # Perform the requested number of traversals.
        for _ in range(self.num_traversals_per_actor):
            state = self.game.new_initial_state()
            # We will traverse for *this* actor's player id (self.player).
            self._traverse_game_tree(state, iteration, self.player)

        return True
    
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
        logits = policy_network(torch.FloatTensor(np.array(info_states)))
        outputs = self.policy_sm(logits)
        loss_strategy = self.loss_policy(iters * outputs, iters * ac_probs)
        loss_strategy.backward()
        # here we need to return the gradients to be aggregated at the parameter server
        return [p.grad.clone() for p in policy_network.parameters()], loss_strategy.detach().numpy()
    
    def _traverse_game_tree(self, state, iteration, player):
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
        return self._traverse_game_tree(state.child(action), iteration, player)
      elif state.current_player() == player:
        sampled_regret = collections.defaultdict(float)
        # Update the policy over the info set & actions via regret matching.
        _, strategy = self._sample_action_from_advantage(state, player)
        for action in state.legal_actions():
          expected_payoff[action] = self._traverse_game_tree(
              state.child(action), iteration, player)
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
        return self._traverse_game_tree(state.child(sampled_action), iteration, player)

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
    
    def advantage_network_step(self, advantage_network):
        """Optimized advantage network training with pre-allocated arrays."""
        # a = time.time()
        # breakpoint()
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
        # a = time.time()
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
        advantage_network.zero_grad()
        advantages_tensor = torch.from_numpy(advantages)
        iters_tensor = torch.from_numpy(np.sqrt(iterations))
        states_tensor = torch.from_numpy(info_states)
        # print("preparing data: ", time.time()-a)
        # a = time.time()
        outputs = advantage_network(states_tensor)
        loss_advantages = self.loss_advantages(iters_tensor * outputs,
                                                iters_tensor * advantages_tensor)
        loss_advantages.backward()
        # get the gradients to be aggregated at the parameter server
        return [p.grad.clone() for p in advantage_network.parameters()], self.player, loss_advantages.detach().numpy()
