import ray
import torch
import torch.nn as nn
import numpy as np
from deep_cfr import MLP
from open_spiel.python import policy
import pickle
from deep_cfr import ReservoirBuffer
from tqdm import tqdm

@ray.remote(num_cpus=2)
class AdvantageLearner:
    def __init__(self, game, memory_capacity, batch_size_advantage, batch_size_strategy, advantage_network_layers, embedding_size, learning_rate, advantage_network_train_steps):
        self.game = game
        self.memory_capacity = memory_capacity
        self.batch_size_advantage = batch_size_advantage
        self.batch_size_strategy = batch_size_strategy
        self.advantage_memory = ReservoirBuffer(memory_capacity)
        self.loss_advantages = nn.MSELoss(reduction="mean")
        self._advantage_network_layers_list = list(advantage_network_layers)
        self.advantage_network = MLP(embedding_size, self._advantage_network_layers_list, # Use the stored list
                self.game.num_distinct_actions())
        self.optimizer_advantage = torch.optim.Adam(self.advantage_network.parameters(), lr=learning_rate)
        self._advantage_network_train_steps = advantage_network_train_steps
        self.unique_info_sets = set()
        self.learning_rate = learning_rate
    def receive_advantage_memories(self, advantage_memories):
        """Receive a batch (list) of AdvantageMemory objects for a player."""
        # Expecting a plain python list; if accidentally passed as ObjectRef, resolve it.
        if isinstance(advantage_memories, ray.ObjectRef):
            advantage_memories = ray.get(advantage_memories)
        self.advantage_memory.add(advantage_memories)
        
    def get_advantage_network(self): 
        return ray.put(self.advantage_network)
    
    def reinitialize_advantage_networks(self):
        """Reinitialize advantage networks for all players"""
        self.advantage_network.reset()
        self.optimizer_advantage = torch.optim.Adam(
            self.advantage_network.parameters(), lr=self.learning_rate)
    
    def learn(self, player):
        for _ in tqdm(range(self._advantage_network_train_steps), desc=f"Training advantage network for player {player}", leave=False):
            if self.batch_size_advantage:
                if self.batch_size_advantage > len(self.advantage_memory):
                    ## Skip if there aren't enough samples
                    return None
                samples = self.advantage_memory.sample(
                    self.batch_size_advantage)
            else:
                samples = self.advantage_memory
            info_states = []
            advantages = []
            iterations = []
            for s in samples:
                info_states.append(s.info_state)
                advantages.append(s.advantage)
                iterations.append([s.iteration])
            # Ensure some samples have been gathered.
            if not info_states:
                return None
            self.optimizer_advantage.zero_grad()
            advantages = torch.FloatTensor(np.array(advantages))
            iters = torch.FloatTensor(np.sqrt(np.array(iterations)))
            outputs = self.advantage_network(torch.FloatTensor(np.array(info_states)))
            loss_advantages = self.loss_advantages(iters * outputs, iters * advantages)
            loss_advantages.backward()
            self.optimizer_advantage.step()
        
        return loss_advantages.detach().numpy()

    def get_advantage_network_states(self):
        """Return a list with the state_dict of each player's advantage network.

        This is useful for saving checkpoints or broadcasting weights without
        sharing the entire model objects across Ray workers.
        """
        return self.advantage_network.state_dict()

@ray.remote(num_cpus=4)
class StrategyLearner:
    def __init__(self, game, memory_capacity, batch_size_strategy, learning_rate, policy_network_train_steps, policy_network_layers, embedding_size):
        self.game = game
        self.memory_capacity = memory_capacity
        self.batch_size_strategy = batch_size_strategy
        self.strategy_memories = ReservoirBuffer(memory_capacity)
        self.loss_strategy = nn.MSELoss(reduction="mean")
        self.policy_sm = nn.Softmax(dim=-1)
        self.policy_network = MLP(embedding_size, policy_network_layers, # Use the stored list
                self.game.num_distinct_actions())
        self.optimizer_strategy = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.policy_network_train_steps = policy_network_train_steps
        self.learning_rate = learning_rate
    def receive_strategy_memories(self, strategy_memories):
        """Receive a batch (list) of StrategyMemory objects."""
        if isinstance(strategy_memories, ray.ObjectRef):
            strategy_memories = ray.get(strategy_memories)
        self.strategy_memories.add(strategy_memories)
            
    def learn(self):
        """Compute the loss over the strategy network.

        Returns:
        (float) The average loss obtained on this batch of transitions or `None`.
        """
        print(f"Training strategy network for {self.policy_network_train_steps} steps")
        for _ in tqdm(range(self.policy_network_train_steps), desc="Training strategy network"):
            if self.batch_size_strategy:
                if self.batch_size_strategy > len(self.strategy_memories):
                ## Skip if there aren't enough samples
                    return None
                samples = self.strategy_memories.sample(self.batch_size_strategy)
            else:
                samples = self.strategy_memories
            info_states = []
            action_probs = []
            iterations = []
            for s in samples:
                info_states.append(s.info_state)
                action_probs.append(s.strategy_action_probs)
                iterations.append([s.iteration])

            self.optimizer_strategy.zero_grad()
            iters = torch.FloatTensor(np.sqrt(np.array(iterations)))
            ac_probs = torch.FloatTensor(np.array(np.squeeze(action_probs)))
            logits = self.policy_network(torch.FloatTensor(np.array(info_states)))
            outputs = self.policy_sm(logits)
            loss_strategy = self.loss_strategy(iters * outputs, iters * ac_probs)
            loss_strategy.backward()
            self.optimizer_strategy.step()

        return loss_strategy.detach().numpy()

    def get_policy_network_state(self):
        """Return the state_dict of the policy network so it can be copied
        into local (non-Ray) models for evaluation or checkpointing.
        """
        return self.policy_network.state_dict()