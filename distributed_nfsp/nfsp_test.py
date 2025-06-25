# Copyright 2024
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training script for PyTorch NFSP agents on Kuhn Poker.

This is a direct analogue of `open_spiel/python/examples/nfsp.py`, but uses the
PyTorch implementation (`open_spiel.python.pytorch.nfsp`).

Run:

    python -m open_spiel.python.examples.kuhn_nfsp_pytorch --num_train_episodes=50000
"""

from absl import app
from absl import flags
from absl import logging
from open_spiel.python import policy
import pyspiel
from open_spiel.python.algorithms import exploitability
from nfsp import NFSP
from parameter_server import ParameterServer
import ray

class NFSPPolicies(policy.Policy):
  """Joint policy constructed from the NFSP agents for evaluation."""

  def __init__(self, env, nfsp_agents, mode):
    game = env.game
    player_ids = [0, 1]
    super().__init__(game, player_ids)
    self._agents = nfsp_agents
    self._mode = mode
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = state.information_state_tensor(cur_player)
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None
    )

    with self._agents[cur_player].temp_mode_as(self._mode):
      p = self._agents[cur_player].step(info_state, is_evaluation=True).probs
    return {action: p[action] for action in legal_actions}
import argparse
import sys

# ---------------------------------------------------------------------------
# Argument parsing (replacement for absl.flags)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Argument parser for NFSP Kuhn Poker training script."
)

parser.add_argument(
    "--num_train_episodes",
    type=int,
    default=int(3e6),
    help="Number of training episodes.",
)
parser.add_argument(
    "--eval_every",
    type=int,
    default=10000,
    help="Episode frequency at which the agents are evaluated.",
)
parser.add_argument(
    "--hidden_layers_sizes",
    type=int,
    nargs="+",
    default=[128, 128],
    help="Number of hidden units in the avg-net and Q-net.",
)
parser.add_argument(
    "--replay_buffer_capacity",
    type=int,
    default=int(2e5),
    help="Size of the replay buffer.",
)
parser.add_argument(
    "--reservoir_buffer_capacity",
    type=int,
    default=int(2e6),
    help="Size of the reservoir buffer.",
)
parser.add_argument(
    "--anticipatory_param",
    type=float,
    default=0.1,
    help="Probability of using the RL best response as episode policy.",
)
parser.add_argument(
    "--num_actors",
    type=int,
    default=4,
    help="Number of actors.",
)
parser.add_argument(
   "--epsilon_start",
   type=float,
   default=0.08,
   help="Starting epsilon for epsilon-greedy exploration.",
)
parser.add_argument(
   "--epsilon_end",
   type=float,
   default=0.001,
   help="Ending epsilon for epsilon-greedy exploration.",
)
parser.add_argument(
   "--learning_rate",
   type=float,
   default=0.001,
   help="Learning rate for the optimizer.",
)
parser.add_argument(
   "--batch_size",
   type=int,
   default=128,
   help="Batch size for the optimizer.",
)

# Parse *known* args so that any additional flags introduced by other
# libraries (e.g. pyspiel, absl) do not trigger a failure.
args, _ = parser.parse_known_args(sys.argv[1:])

num_players = 2

game = pyspiel.load_game("kuhn_poker")
info_state_size = game.information_state_tensor_size()
num_actions = game.num_distinct_actions()

hidden_layers_sizes = [int(x) for x in args.hidden_layers_sizes]
dqn_kwargs = {
    "replay_buffer_capacity": args.replay_buffer_capacity,
    "epsilon_decay_duration": args.num_train_episodes,
    "epsilon_start": 0.08,
    "epsilon_end": 0.001,
}
# initialize ray if not initialized
if not ray.is_initialized():
    # Initialize Ray in a more debug-friendly configuration.
    # `local_mode=True` runs all remote calls synchronously in the
    # local process, which makes it much easier to step through with
    # a debugger and to read stack traces.  We also forward worker
    # logs to the driver and raise the logging verbosity.
    import logging
    ray.init(
        local_mode=True,            # Execute tasks/actors locally for step-through debugging
        log_to_driver=True,         # Stream logs from workers to the driver
        logging_level=logging.INFO  # Increase log verbosity for better traceability
    )
parameter_server = ParameterServer.remote(
    game,
    info_state_size,
    num_actions,
    hidden_layers_sizes,
    args.batch_size,
    args.replay_buffer_capacity,
    args.reservoir_buffer_capacity,
    args.anticipatory_param,
    args.num_train_episodes,
    args.eval_every,
    args.epsilon_start,
    args.epsilon_end,
    args.num_train_episodes,
    args.learning_rate,
    args.anticipatory_param,
    args.num_actors,
)
parameter_server.start.remote()