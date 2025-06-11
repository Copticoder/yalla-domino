import ray
import collections
import torch
import numpy as np
from deep_cfr import MLP
from tqdm import tqdm
import wandb
from actor import DeepCFRActor
from parameter_server import ParameterServer
from evaluator import Evaluator

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
                 use_wandb: bool = False,
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
        self._optimizer_policy = torch.optim.Adam(
            self._policy_network.parameters(), lr=learning_rate)
        self._policy_network_ref = ray.put(self._policy_network)
        self.advantage_network_train_steps = advantage_network_train_steps
        
        wandb_config = {"lr": self.learning_rate, "batch_size": self.batch_size_advantage, "memory_capacity": self.memory_capacity, "policy_network_train_steps": self.policy_network_train_steps, "advantage_network_train_steps": self.advantage_network_train_steps, "num_actors": self.num_actors, "num_traversals": self.num_traversals, "num_iterations": self.num_iterations, "evaluation_interval": self.evaluation_interval, "num_players": self.game.num_players(), "policy_network_layers": self.policy_network_layers, "advantage_network_layers": self.advantage_network_layers, "reinitialize_advantage_networks": self.reinitialize_advantage_networks}   
        run = None
        if use_wandb:
            run = wandb.init(project="deep_cfr_ray", config=wandb_config)
        
        self.evaluator = Evaluator(self.game, self._policy_network, self.evaluation_interval, self.policy_network_train_steps, self.num_actors, self.learning_rate, run)
    def _initialize_actors(self, num_actors):
        """Initialize actors with current network parameters"""
        # Calculate traversals per actor
        self.num_traversals_per_actor = max(1, self.num_traversals // (self.num_actors-2))
        self.actors = []
        # create a list of half for player 1 and the other half for player 2
        for player in range(self.game.num_players()):
            self.actors += [DeepCFRActor.options(name=f"actor_{player}_{i}", lifetime="detached", namespace="deep_cfr").remote(self.game, self.num_traversals_per_actor, self.memory_capacity, self.batch_size_advantage, self.batch_size_strategy, player) for i in range(num_actors//self.game.num_players())]

    def solve(self):
        """Solution logic for Deep CFR."""
        advantage_losses = collections.defaultdict(list)
        unique_info_states = set()
        self._initialize_actors(self.num_actors - 2)
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
            # from every actor, we need to get the unique info states
            unique_info_states = unique_info_states.union(*output)
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
                policy_losses = self.evaluator.evaluate(len(unique_info_states))

        return self._policy_network, advantage_losses, policy_losses, unique_info_states
    

    