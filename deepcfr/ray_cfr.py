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
    # number of cpus
    game = pyspiel.load_game("leduc_poker")
    # Placement group reserves CPU resources exclusively for traversal actors.
    # Learner workers are left to use any remaining cluster resources on demand.
    placement_group = ray.util.placement_group([
        {"CPU": 12},  # Bundle 0: Actors (traversal workers)
    ])

    # Wait until the placement group resources are ready.
    ray.get(placement_group.ready())

    # Number of traversal actors that will use the first bundle (12 CPUs).
    num_actors = 12

    solver = Orchestrator(
    game,
    policy_network_layers=(64,64,64),
    advantage_network_layers=(64,64,64),
    num_iterations=300,
    num_traversals=1500,
    reinitialize_advantage_networks=True,
    learning_rate=1e-3,
    batch_size_advantage=256,
    batch_size_strategy=256,
    memory_capacity=int(1e5),
    policy_network_train_steps=5000,
    advantage_network_train_steps=750,
    evaluation_interval=10,
    num_actors=num_actors,
    placement_group=placement_group,
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
