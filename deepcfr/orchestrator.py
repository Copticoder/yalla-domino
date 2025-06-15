import ray
import collections
import torch
import numpy as np
from deep_cfr import MLP
from tqdm import tqdm
import wandb
from actor import DeepCFRActor
from evaluator import Evaluator
from learners import AdvantageLearner, StrategyLearner
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
                 placement_group=None,
                 use_wandb: bool = False,
                 training_mode: bool = True,
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
        self.placement_group = placement_group
        # Store training steps for policy network
        self.policy_network_train_steps = policy_network_train_steps
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
        self.advantage_network_train_steps = advantage_network_train_steps
        self.training_mode = training_mode
        
        self.advantage_learners = [AdvantageLearner.options(**{
            "name": f"advantage_learner_{player}_{{idx}}", 
            "lifetime": "detached", 
            "namespace": "deep_cfr",
        }).remote(
            self.game,
            self.memory_capacity,
            self.batch_size_advantage,
            self.batch_size_strategy,
            self.advantage_network_layers,
            self._embedding_size,
            self.learning_rate,
            self.advantage_network_train_steps,
        ) for player in range(self.game.num_players())]
        
        self.strategy_learner = StrategyLearner.remote(
            self.game,
            self.memory_capacity,
            self.batch_size_strategy,
            self.learning_rate,
            self.policy_network_train_steps,
            self.policy_network_layers,
            self._embedding_size,
        )
        
        wandb_config = {"lr": self.learning_rate, "batch_size": self.batch_size_advantage, "memory_capacity": self.memory_capacity, "policy_network_train_steps": self.policy_network_train_steps, "advantage_network_train_steps": self.advantage_network_train_steps, "num_actors": self.num_actors, "num_traversals": self.num_traversals, "num_iterations": self.num_iterations, "evaluation_interval": self.evaluation_interval, "num_players": self.game.num_players(), "policy_network_layers": self.policy_network_layers, "advantage_network_layers": self.advantage_network_layers, "reinitialize_advantage_networks": self.reinitialize_advantage_networks}   
        run = None
        if use_wandb:
            run = wandb.init(project="deep_cfr_ray", config=wandb_config)
        
        self.evaluator = Evaluator(self.game, self._policy_network, self.evaluation_interval, self.policy_network_train_steps, self.num_actors, self.learning_rate, run, self.strategy_learner)
    
    def _initialize_actors(self, num_actors):
        """Initialize actors with current network parameters"""
        # Calculate traversals per actor
        self.num_traversals_per_actor = max(1, self.num_traversals // max(1, num_actors))
        self.actors = []
        # create a list of half for player 1 and the other half for player 2
        for player in range(self.game.num_players()):
            actor_opts = {
                "name": f"actor_{player}_{{idx}}",
                "lifetime": "detached",
                "namespace": "deep_cfr",
                "placement_group": self.placement_group,
                "placement_group_bundle_index": 0,
            } if self.placement_group else {"name": f"actor_{player}_{{idx}}", "lifetime": "detached", "namespace": "deep_cfr"}

            self.actors += [DeepCFRActor.options(**{**actor_opts, "name": actor_opts["name"].format(idx=i)}).remote(
                self.game,
                self.num_traversals_per_actor,
                self.memory_capacity,
                self.batch_size_advantage,
                self.batch_size_strategy,
                player,
                self.advantage_learners[player],
                self.strategy_learner,
            ) for i in range(num_actors // self.game.num_players())]

    def solve(self):
        """Solution logic for Deep CFR."""
        advantage_losses = collections.defaultdict(list)
        unique_info_states = set()
        self._initialize_actors(self.num_actors)
        if self.training_mode:
            print("=="*10, "Training mode", "=="*10)
            for i in tqdm(range(self.num_iterations), desc="CFR Iterations"):
                # Parallel traversals
                traversal_tasks = []
                for player in range(self.game.num_players()):
                    actors = [ray.get_actor(f"actor_{player}_{i}", namespace="deep_cfr")
                            for i in range(self.num_actors // self.game.num_players())]

                    # Broadcast current advantage networks to all actors.
                    all_networks = ray.get(self.advantage_learners[player].get_advantage_network.remote())
                    traversal_tasks += [actor.batch_traverse_solve_game.remote(i, all_networks)
                                        for actor in actors]
                    

                # Wait for all traversals to complete and collect results using ray.wait
                outputs = []
                remaining_tasks = traversal_tasks
                pbar_traverse = tqdm(total=len(remaining_tasks), desc=f"Traversals iteration {i}")
                while remaining_tasks:
                    done, remaining_tasks = ray.wait(remaining_tasks, num_returns=min(20, len(remaining_tasks)))
                    outputs.extend(ray.get(done))
                    pbar_traverse.update(len(done))
                pbar_traverse.close()

                # NOTE: Actors currently do not return the visited information
                # states. If detailed visitation statistics are required, the
                # actors can be extended to return them. For now we keep count of
                # traversals only.

                # ------------------------------------------------------------------
                # After traversal data has been collected, train the networks.
                # ------------------------------------------------------------------
                # Reinitialize advantage networks (optional)
                if self.reinitialize_advantage_networks:
                    self.advantage_learners[player].reinitialize_advantage_networks.remote()

                # Train advantage networks for each player and collect losses.
                advantage_losses_iter = []
                # Train advantage networks for all players in parallel
                advantage_losses_iter = [self.advantage_learners[player].learn.remote(player) 
                                    for player in range(self.game.num_players())]

                advantage_losses_values = ray.get(advantage_losses_iter)    
                for p, l in enumerate(advantage_losses_values):
                    if l is not None:
                        advantage_losses[p].append(l)

                # Logging
                for player in range(self.game.num_players()):
                    if advantage_losses[player]:
                        print(f"Advantage loss for player {player}: {advantage_losses[player][-1]}")

                if i % self.evaluation_interval == 0:
                    policy_losses = self.evaluator.evaluate(len(unique_info_states))
                # save the advantage memories and the policy network
                for player in range(self.game.num_players()):
                    ray.get(self.advantage_learners[player].save_memories.remote())
                ray.get(self.strategy_learner.save_memories.remote())
                ray.get(self.strategy_learner.save_network.remote())
                ray.get(self.advantage_learners[player].save_network.remote())
                
            # Train strategy / policy network
            strategy_loss = ray.get(self.strategy_learner.learn.remote())
            if strategy_loss is not None:
                print(f"Strategy network loss: {strategy_loss}")
            return self._policy_network, advantage_losses, policy_losses, unique_info_states
        else:
            print("=="*10, "Evaluation mode", "=="*10)
            # load latest policy network
            self.strategy_learner.load_network.remote()
            self.evaluator.evaluate(len(unique_info_states))
    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def action_probabilities(self, state):
        """Delegate to the evaluator's current policy network."""
        return self.evaluator.action_probabilities(state)

