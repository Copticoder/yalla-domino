import ray
import torch
import torch.nn as nn
import numpy as np
from deep_cfr import MLP
from open_spiel.python import policy
import pickle
from deep_cfr import ReservoirBuffer
from tqdm import tqdm

@ray.remote(num_cpus=4, num_gpus=0.25)
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
        self.unique_info_sets = {}
        self.learning_rate = learning_rate
    
    def receive_advantage_memories(self, advantage_memories):
        """Receive a batch (list) of AdvantageMemory objects for a player."""
        # Expecting a plain python list; if accidentally passed as ObjectRef, resolve it.
        if isinstance(advantage_memories, ray.ObjectRef):
            advantage_memories = ray.get(advantage_memories)
        self.advantage_memory.add(advantage_memories)
    
    
    def get_advantage_network(self):
        # Always move the network to the CPU before broadcasting it. This
        # prevents Ray workers that do not have a GPU from trying to
        # deserialize CUDA tensors, which leads to `RuntimeError: Attempting
        # to deserialize object on a CUDA device but torch.cuda.is_available()
        # is False`.
        self.advantage_network.to(torch.device("cpu"))
        return ray.put(self.advantage_network)
    
    def reinitialize_advantage_networks(self):
        """Reinitialize advantage networks for all players"""
        self.advantage_network.reset()
        self.optimizer_advantage = torch.optim.Adam(
            self.advantage_network.parameters(), lr=self.learning_rate)
    
    def learn(self, player):
        # put the advantage network on the gpu
        self.advantage_network.to(torch.device("cuda"))
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
            advantages = torch.FloatTensor(np.array(advantages)).to(torch.device("cuda"))
            iters = torch.FloatTensor(np.sqrt(np.array(iterations))).to(torch.device("cuda"))
            outputs = self.advantage_network(torch.FloatTensor(np.array(info_states)).to(torch.device("cuda")))
            loss_advantages = self.loss_advantages(iters * outputs, iters * advantages)
            loss_advantages.backward()
            self.optimizer_advantage.step()
        # Move the trained network back to CPU so that subsequent
        # serialisation does not embed CUDA tensors.
        self.advantage_network.to(torch.device("cpu"))
        return loss_advantages.cpu().detach().numpy()

    def get_advantage_network_states(self):
        """Return a list with the state_dict of each player's advantage network.

        This is useful for saving checkpoints or broadcasting weights without
        sharing the entire model objects across Ray workers.
        """
        return self.advantage_network.state_dict()
    def save_memories(self):
        """Save the memories to a file."""
        with open("advantage_memories.pkl", "wb") as f:
            pickle.dump(self.advantage_memory, f)
    def load_memories(self):
        """Load the memories from a file."""
        with open("advantage_memories.pkl", "rb") as f:
            self.advantage_memory = pickle.load(f)
    def save_network(self):
        """Save the network to a file."""
        torch.save(self.advantage_network.state_dict(), "advantage_network.pth")
    def load_network(self):
        """Load the network from a file."""
        self.advantage_network.load_state_dict(torch.load("advantage_network.pth"))
@ray.remote(num_gpus=0.5, num_cpus=8)
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
        self.unique_info_sets = {}
    def receive_strategy_memories(self, strategy_memories):
        """Receive a batch (list) of StrategyMemory objects."""
        if isinstance(strategy_memories, ray.ObjectRef):
            strategy_memories = ray.get(strategy_memories)
        self.strategy_memories.add(strategy_memories)
    
    def get_num_unique_info_sets(self):
        for s in self.strategy_memories:
            if tuple(s.info_state) not in self.unique_info_sets:
                self.unique_info_sets[tuple(s.info_state)] = 1
        return len(self.unique_info_sets)
    
    def learn(self):
        """Compute the loss over the strategy network.

        Returns:
        (float) The average loss obtained on this batch of transitions or `None`.
        """
        self.policy_network.to(torch.device("cuda"))
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
            iters = torch.FloatTensor(np.sqrt(np.array(iterations))).to(torch.device("cuda"))
            ac_probs = torch.FloatTensor(np.array(np.squeeze(action_probs))).to(torch.device("cuda"))
            logits = self.policy_network(torch.FloatTensor(np.array(info_states)).to(torch.device("cuda")))
            outputs = self.policy_sm(logits)
            loss_strategy = self.loss_strategy(iters * outputs, iters * ac_probs)
            loss_strategy.backward()
            self.optimizer_strategy.step()
        self.policy_network.to(torch.device("cpu"))
        return loss_strategy.cpu().detach().numpy()

    def get_policy_network_state(self):
        """Return the state_dict of the policy network with CPU tensors.

        Having all tensors on the CPU avoids deserialization errors when the
        caller is running in an environment without CUDA.
        """
        # Ensure weights reside on CPU first.
        self.policy_network.to(torch.device("cpu"))
        # Clone tensors onto CPU explicitly.
        return {k: v.cpu() for k, v in self.policy_network.state_dict().items()}
    def save_memories(self):
        """Save the memories to a file."""
        with open("strategy_memories.pkl", "wb") as f:
            pickle.dump(self.strategy_memories, f)
    def load_memories(self):
        """Load the memories from a file."""
        with open("strategy_memories.pkl", "rb") as f:
            self.strategy_memories = pickle.load(f)
    def save_network(self):
        """Save the network to a file."""
        torch.save(self.policy_network.state_dict(), "policy_network.pth")
    def load_network(self):
        """Load the network from a file."""
        self.policy_network.load_state_dict(torch.load("policy_network.pth"))