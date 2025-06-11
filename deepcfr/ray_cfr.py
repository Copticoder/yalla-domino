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
        runtime_env={"env_vars": {"RAY_DEBUG": "1"}},
        ignore_reinit_error=True,
    )
    game = pyspiel.load_game("python_block_dominoes")
    # Get number of available CPUs for Ray actors
    num_cpus = ray.cluster_resources()['CPU']
    # Leave 1 CPU for the main process
    num_actors = max(1, int(num_cpus) - 1)

    solver = Orchestrator(
    game,
    policy_network_layers=(256,64,64),
    advantage_network_layers=(256,64,64),
    num_iterations=300,
    num_traversals=1000,
    reinitialize_advantage_networks=True,
    learning_rate=1e-3,
    batch_size_advantage=384,
    batch_size_strategy=384,
    memory_capacity=1e6,
    policy_network_train_steps=1,
    advantage_network_train_steps=1,
    evaluation_interval=5,
    num_actors=num_actors,
    use_wandb=False
    )
    _, advantage_losses, policy_loss = solver.solve()
    
    for player, losses in list(advantage_losses.items()):
        print("Advantage for player:", player,
                      losses[:2] + ["..."] + losses[-2:])
        
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
