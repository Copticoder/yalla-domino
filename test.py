import itertools
from open_spiel.python import policy as policy_module
from open_spiel.python.algorithms import exploitability
from deep_cfr import DeepCFRSolver
import pyspiel

def print_policy(policy):
  for state, probs in zip(itertools.chain(*policy.states_per_player),
                          policy.action_probability_array):
    print(f'{state:6}   p={probs}')


def main():
  game = pyspiel.load_game('leduc_poker')
  solver = DeepCFRSolver(
        game,
        policy_network_layers=(64,64,64),
        advantage_network_layers=(64,64,64),
        num_iterations=101,
        reinitialize_advantage_networks=True,
        num_traversals=10000,
        learning_rate=1e-3,
        batch_size_advantage=2048,
        batch_size_strategy=2048,
        memory_capacity=int(1e6),
        policy_network_train_steps=5000,
        advantage_network_train_steps=750,
) 
  import time
  start_time = time.time()
  _, advantage_losses, policy_loss = solver.solve()
  end_time = time.time()
  print(f"Time taken: {end_time - start_time:.2f} seconds")
  for player, losses in list(advantage_losses.items()):
    print("Advantage for player:", player,
                  losses[:2] + ["..."] + losses[-2:])
    print("Advantage Buffer Size for player", player,
                  len(solver.advantage_buffers[player]))
  print("Strategy Buffer Size:",
                len(solver.strategy_buffer))
  print("Final policy loss:", policy_loss)
  # Get Deep CFR policy
  policy = policy_module.tabular_policy_from_callable(game, solver.action_probabilities)
  conv = exploitability.nash_conv(game, policy)
  print("Deep CFR - NashConv:", conv)

#   np.set_printoptions(precision=3, suppress=True, floatmode='fixed')
#   print_policy(policy)
#   test random policy
#   random_policy = policy_module.TabularPolicy(game)
#   conv = exploitability.nash_conv(game, random_policy)
#   print("Random policy - NashConv:", conv)


if __name__ == "__main__":
  main()

