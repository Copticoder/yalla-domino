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
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
import torch
import pyspiel
from torch import nn
import torch.nn.functional as F
import math
from typing import Sequence, List
from open_spiel.python.pytorch import nfsp
from scipy import stats
import nfsp
FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_episodes", int(3e6),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 10000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_list("hidden_layers_sizes", [128],
                 "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Probability of using the RL best response as episode policy.")
flags.DEFINE_integer("batch_size", 256,
                     "Batch size for the DQN.")
class SonnetLinear(nn.Module):
  """A Sonnet linear module.

  Always includes biases and only supports ReLU activations.
  """

  def __init__(self, in_size, out_size, activate_relu=True):
    """Creates a Sonnet linear layer.

    Args:
      in_size: (int) number of inputs
      out_size: (int) number of outputs
      activate_relu: (bool) whether to include a ReLU activation layer
    """
    super(SonnetLinear, self).__init__()
    self._activate_relu = activate_relu
    stddev = 1.0 / math.sqrt(in_size)
    mean = 0
    lower = (-2 * stddev - mean) / stddev
    upper = (2 * stddev - mean) / stddev
    # Weight initialization inspired by Sonnet's Linear layer,
    # which cites https://arxiv.org/abs/1502.03167v3
    # pytorch default: initialized from
    # uniform(-sqrt(1/in_features), sqrt(1/in_features))
    self._weight = nn.Parameter(
        torch.Tensor(
            stats.truncnorm.rvs(
                lower, upper, loc=mean, scale=stddev, size=[out_size,
                                                            in_size])))
    self._bias = nn.Parameter(torch.zeros([out_size]))

  def forward(self, tensor):
    y = F.linear(tensor, self._weight, self._bias)
    return F.relu(y) if self._activate_relu else y


class BR_MLP(nn.Module):
  """A simple network built from nn.linear layers."""

  def __init__(self,
               input_size,
               hidden_sizes,
               output_size,
               activate_final=False):
    """Create the MLP.

    Args:
      input_size: (int) number of inputs
      hidden_sizes: (list) sizes (number of units) of each hidden layer
      output_size: (int) number of outputs
      activate_final: (bool) should final layer should include a ReLU
    """

    super(BR_MLP, self).__init__()
    self._layers = []
    # Hidden layers
    for size in hidden_sizes:
      self._layers.append(SonnetLinear(in_size=input_size, out_size=size))
      input_size = size
    # Output layer
    self._layers.append(
        SonnetLinear(
            in_size=input_size,
            out_size=output_size,
            activate_relu=activate_final))

    self.model = nn.ModuleList(self._layers)

  def forward(self, x):
    for layer in self.model:
      x = layer(x)
    return x
class AVG_MLP(nn.Module):
  """Simple MLP identical to the one used in the DQN PyTorch agent."""

  def __init__(self, in_size: int, hidden_sizes: Sequence[int], out_size: int):
    super().__init__()
    sizes = list(hidden_sizes) + [out_size]
    layers: List[nn.Module] = []
    for hs in sizes[:-1]:
      layers.append(nn.Linear(in_size, hs))
      layers.append(nn.ReLU())
      in_size = hs
    layers.append(nn.Linear(in_size, sizes[-1]))  # last layer – no activation
    self._model = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
    return self._model(x)
class NFSPPolicies(policy.Policy):
  """Joint policy constructed from the NFSP agents for evaluation."""

  def __init__(self, env, nfsp_actor):
    game = env
    player_ids = [0, 1]
    super().__init__(game, player_ids)
    self._actor = nfsp_actor
    
  def action_probabilities(self, state):
    cur_player = state.current_player()

    legal_actions = state.legal_actions()
    # Ask the NFSP actor for an action (deterministic in evaluation mode).
    chosen_action, probs = self._actor.step(state, cur_player, is_evaluation=True)

    # Build a full probability distribution over all legal actions.
    return {a: float(probs[a]) for a in legal_actions}

def main(_):
  game = "leduc_poker"
  num_players = 2

  env = pyspiel.load_game(game)
  info_state_size = env.information_state_tensor_size()
  num_actions = env.num_distinct_actions()
  hidden_layers_sizes = [int(x) for x in FLAGS.hidden_layers_sizes]
  learning_rate = 0.01
  q_networks = [BR_MLP(info_state_size, hidden_layers_sizes, num_actions) for _ in range(num_players)]
  q_net_optimizers = [torch.optim.SGD(q_network.parameters(), lr=learning_rate) for q_network in q_networks]
  avg_networks = [AVG_MLP(info_state_size, hidden_layers_sizes, num_actions) for _ in range(num_players)]
  avg_net_optimizers = [torch.optim.SGD(avg_network.parameters(), lr=learning_rate) for avg_network in avg_networks]

  actor = nfsp.NFSP(
          game = env,
          num_players = num_players,
          num_actions = num_actions,
          replay_buffer_capacity = FLAGS.replay_buffer_capacity,
          reservoir_buffer_capacity = FLAGS.reservoir_buffer_capacity,
          anticipatory_param = FLAGS.anticipatory_param,
          batch_size=FLAGS.batch_size,
          min_buffer_size_to_learn=FLAGS.batch_size,
          q_networks=q_networks,
          q_net_optimizers=q_net_optimizers,
          avg_networks=avg_networks,
          avg_net_optimizers=avg_net_optimizers,
      )

  eval_policy_avg = NFSPPolicies(env, actor)

  for ep in range(FLAGS.num_train_episodes):
    if (ep + 1) % FLAGS.eval_every == 0:
      losses = [actor.get_loss(player_id) for player_id in range(num_players)]
      logging.info("Episode %s - Losses %s", ep + 1, losses)
      expl = exploitability.exploitability(env, eval_policy_avg)
      logging.info("Episode %s - Exploitability (avg policy): %.4f", ep + 1, expl)
      logging.info("----------------------------------------------------")
    
    actor.traverse_game()
if __name__ == "__main__":
  app.run(main) 