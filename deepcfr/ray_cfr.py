import itertools
from open_spiel.python import policy as policy_module
from open_spiel.python.algorithms import exploitability
import ray
import pyspiel
import numpy as np
from orchestrator import Orchestrator


if __name__ == "__main__":
    try:
        ray.shutdown()
    except:
        pass
    
    ray.init(
        namespace="deep_cfr",
        ignore_reinit_error=True,
    )
    # number of cpus
    game_name = "python_block_dominoes"
    game = pyspiel.load_game(game_name)

    # Number of traversal actors that will use the first bundle (12 CPUs).
    num_actors = 12

    solver = Orchestrator(
    game,
    policy_network_layers=(256,256,128,64),
    advantage_network_layers=(256,256,128,64),
    num_iterations=300,
    num_traversals=20000,
    reinitialize_advantage_networks=True,
    learning_rate=1e-3,
    batch_size_advantage=20000,
    batch_size_strategy=20000,
    memory_capacity=int(40e6),
    policy_network_train_steps=16000,
    advantage_network_train_steps=16000,
    evaluation_interval=10,
    num_actors=num_actors,
    use_wandb=True,
    training_mode=True
    )
    solver.solve()
    
    if game_name != "python_block_dominoes":
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
