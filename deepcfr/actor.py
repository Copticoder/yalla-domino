import ray
import collections
import torch
import torch.nn as nn
import numpy as np
from deep_cfr import ReservoirBuffer, StrategyMemory, AdvantageMemory
@ray.remote
class DeepCFRActor:
    def __init__(self, game, num_traversals_per_actor, memory_capacity, batch_size_advantage, batch_size_strategy, player, advantage_learner, strategy_learner):
        self.game = game
        self._num_actions = game.num_distinct_actions()
        self.num_traversals_per_actor = num_traversals_per_actor
        self.num_distinct_actions = game.num_distinct_actions()
        self._embedding_size = len(game.new_initial_state().information_state_tensor(0))
        # Define advantage network, loss & memory. (One per player)
        self._advantage_memory = []
        self._strategy_memories = []
        # Define strategy network, loss & memory.
        self.memory_capacity = memory_capacity
        self.loss_advantages = nn.MSELoss(reduction="mean")
        self.batch_size_advantage = batch_size_advantage
        self.batch_size_strategy = batch_size_strategy
        self.policy_sm = nn.Softmax(dim=-1)
        self.loss_policy = nn.MSELoss()
        self.player = player
        self.advantage_learner = advantage_learner
        self.strategy_learner = strategy_learner
    def batch_traverse_solve_game(self, iteration, advantage_network):
        """Perform multiple traversals and collect data locally before adding to shared memory.

        Args:
            iteration (int): Current training iteration.
            advantage_networks (List[nn.Module]): A list containing one advantage
                network per player. The actor will look up the correct network
                based on the player index encountered during traversal.
        """
        self.advantage_network = ray.get(advantage_network) if isinstance(advantage_network, ray.ObjectRef) else advantage_network

        # Perform the requested number of traversals.
        for _ in range(self.num_traversals_per_actor):
            state = self.game.new_initial_state()
            # We will traverse for *this* actor's player id (self.player).
            self._traverse_game_tree(state, iteration, self.player)
        
        # Send collected memories to the learners for training
        self.advantage_learner.receive_advantage_memories.remote(self._advantage_memory)
        self.strategy_learner.receive_strategy_memories.remote(self._strategy_memories)
        self._advantage_memory = []
        self._strategy_memories = []
        return True
    
    
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
        self._advantage_memory.append(
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
        self._strategy_memories.append(
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