import argparse
import time
import statistics

import pyspiel
from deep_cfr import DeepCFRSolver


def benchmark(game_name: str, traversals: int = 100, player: int = 0):
    """Benchmark `_traverse_game_tree` on the provided game.

    Args:
        game_name (str): Name of the OpenSpiel game to load.
        traversals (int): How many traversals to run.
        player (int): Player index to traverse for.
    """
    game = pyspiel.load_game(game_name)

    # Instantiate solver with minimal work besides traversal.
    solver = DeepCFRSolver(game, num_iterations=1, num_traversals=1)

    times = []
    for _ in range(traversals):
        state = game.new_initial_state()
        t0 = time.perf_counter()
        solver._traverse_game_tree(state, player)
        times.append(time.perf_counter() - t0)

    mean_t = statistics.mean(times)
    median_t = statistics.median(times)
    print(f"Benchmark results for `{game_name}` over {traversals} traversals (player {player}):")
    print(f"  mean   : {mean_t:.6f} s")
    print(f"  median : {median_t:.6f} s")
    print(f"  min    : {min(times):.6f} s")
    print(f"  max    : {max(times):.6f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark _traverse_game_tree speed.")
    parser.add_argument("--game", default="draw_dominoes", help="OpenSpiel game name")
    parser.add_argument("--traversals", type=int, default=1, help="Number of traversals to run")
    parser.add_argument("--player", type=int, default=0, help="Player index to traverse for")
    args = parser.parse_args()

    benchmark(args.game, args.traversals, args.player) 