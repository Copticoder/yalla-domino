import os
import ray
import torch
import torch.nn as nn
import numpy as np
from deep_cfr import MLP
from open_spiel.python import policy
import pickle

@ray.remote
class ParameterServer():
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
    def put_network(self):
        return ray.put(self._advantage_network)
    def clear_advantage_buffers(self):
        self._advantage_memories.clear()
        
    def aggregate_gradients(self, gradient_buffer):
        grouped_by_param = zip(*gradient_buffer)
        
        averaged_gradients = []
        # breakpoint()
        for group in grouped_by_param:
            averaged_gradients.append(torch.stack(group).mean(dim=0))
        for p, g in zip(self._advantage_network.parameters(), averaged_gradients):
            p.grad = g
        self._optimizer_advantage.step()
        self._optimizer_advantage.zero_grad()
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
    
    def save_advantage_network(self, path):
        torch.save(self._advantage_network.state_dict(), path)
        return True

    def load_advantage_network(self, path):
        self._advantage_network.load_state_dict(torch.load(path))
        return True