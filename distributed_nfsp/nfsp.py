# Copyright 2024.  Inspired by JAX implementation of NFSP.
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

"""Neural Fictitious Self-Play (NFSP) agent implemented in PyTorch.

This implementation follows the logic of the original JAX version in
`open_spiel/python/jax/nfsp.py`, replacing the functional-style JAX
operations with imperative PyTorch code.

Usage example: see `open_spiel/python/examples/kuhn_nfsp.py` but replace the
`jax.nfsp.NFSP` import with `pytorch.nfsp.NFSP`.
"""

from __future__ import annotations

import collections
import contextlib
import enum
import os
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import ray
from open_spiel.python import rl_agent
from dqn import DQN
from open_spiel.python.utils.reservoir_buffer import ReservoirBuffer

Transition = collections.namedtuple(
    "Transition", "info_state action_probs legal_actions_mask")

MODE = enum.Enum("mode", "best_response average_policy")


class MLP(nn.Module):
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

@ray.remote(num_cpus=1)
class NFSP(rl_agent.AbstractAgent):
  """NFSP Agent implementation in PyTorch."""

  def __init__(
      self,
      game,
      num_players: int,
      state_representation_size: int,
      num_actions: int,
      reservoir_buffer_capacity: int,
      anticipatory_param: float,
      batch_size: int = 128,
      min_buffer_size_to_learn: int = 1000,
      learn_every: int = 64,
      avg_net_optimizers: torch.optim.Optimizer = None,
      avg_networks: nn.Module = None,
      q_net_optimizers: torch.optim.Optimizer = None,
      q_networks: nn.Module = None,
      **kwargs,
  ) -> None:
    self._game = game
    self._num_actions = num_actions
    self._batch_size = batch_size
    self._learn_every = learn_every
    self._anticipatory_param = anticipatory_param
    self._min_buffer_size_to_learn = min_buffer_size_to_learn

    self._reservoir_buffers = [ReservoirBuffer(reservoir_buffer_capacity) for _ in range(num_players)]
    self._prev_state = None
    self._prev_action = None

    # Step counter to keep track of learning.
    self._step_counters = [0 for _ in range(num_players)]

    self._rl_agents = [DQN(
        player_id,
        state_representation_size,
        num_actions,
        optimizer=q_net_optimizers[player_id],
        q_network=q_networks[player_id],
        **kwargs,
    ) for player_id in range(num_players)]

    # Keep track of the last training loss achieved in an update step.
    self._last_rl_loss_value = lambda: self._rl_agent.loss
    self._last_sl_loss_value = None

    # Average policy network.
    self._avg_networks = avg_networks

    # Optimizer for supervised learning (SL) network.
    self._optimizers = avg_net_optimizers

    self._savers = [
        ("q_network", self._rl_agents),
        ("avg_network", self._avg_networks),
    ]

    # Initialize episode policy.
    self._sample_episode_policy()
    
  def update_networks(self, q_networks, avg_networks, q_net_optimizers, avg_net_optimizers):
    for i in range(self._num_players):
      self._rl_agents[i].q_network.load_state_dict(q_networks[i])
      self._avg_networks[i].load_state_dict(avg_networks[i])
      self._q_net_optimizers[i].load_state_dict(q_net_optimizers[i])
      self._avg_net_optimizers[i].load_state_dict(avg_net_optimizers[i])
  
  # ---------------------------------------------------------------------------
  # Acting
  # ---------------------------------------------------------------------------
  
  def traverse_game(self):
    state = self._game.reset()
    while True:
      if state.chance_node():
        legal_actions = state.legal_actions()
        action = np.random.choice(legal_actions)
        state = self._game.step([action])
      else:
        action = self.step(state)
        if state.is_terminal():
          break
        state = state.apply_action(action)
    return True  
  def _sample_episode_policy(self):
    if np.random.rand() < self._anticipatory_param:
      self._mode = MODE.best_response
    else:
      self._mode = MODE.average_policy

  def _act(self, info_state, legal_actions, player_id):
    info_state_t = torch.Tensor(np.reshape(info_state, [1, -1]))
    with torch.no_grad():
      logits = self._avg_networks[player_id](info_state_t)
      action_probs_t = F.softmax(logits, dim=1)
    action_probs = action_probs_t.squeeze(0).cpu().numpy()

    # Remove illegal actions, renormalize probs.
    probs = np.zeros(self._num_actions)
    probs[legal_actions] = action_probs[legal_actions]
    probs_sum = probs.sum()
    if probs_sum == 0:
      # If numerical instabilities zero out all, fall back to uniform over legal.
      probs[legal_actions] = 1.0 / len(legal_actions)
    else:
      probs /= probs_sum
    action = np.random.choice(len(probs), p=probs)
    return action, probs

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  @property
  def mode(self):
    return self._mode

  @property
  def loss(self):
    return (self._last_sl_loss_value, self._last_rl_loss_value())
  # ---------------------------------------------------------------------------
  # RL-Agent compatible interface
  # ---------------------------------------------------------------------------
  def step(self, state, is_evaluation: bool = False):
    """Returns the action to be taken and updates the networks if needed."""
    if self._mode == MODE.best_response:
      action, probs = self._rl_agents[state.current_player()].step(state, is_evaluation)
      if not is_evaluation and not state.is_terminal():
        self._add_transition(state, probs)

    elif self._mode == MODE.average_policy:
      if not state.is_terminal():
        info_state = state.info_state_tensor()
        legal_actions = state.legal_actions()
        action, _ = self._act(info_state, legal_actions, state.current_player())

      # Feed experience to RL agent.
      if self._prev_state and not is_evaluation:
        self._rl_agents[state.current_player()].add_transition(self._prev_state, self._prev_action, state)
        
      if state.is_terminal():
        self._prev_state = None
        self._prev_action = None
        return None
      else:
        self._prev_state = state
        self._prev_action = action
    else:
      raise ValueError(f"Invalid mode ({self._mode})")
    return action
  
  def learn_br_rl(self, player_id):    
    gradients_sl, self._last_sl_loss_value = self._learn(player_id)
    gradients_rl, self._last_rl_loss_value = self._rl_agents[player_id].learn()
    return gradients_sl, gradients_rl
  
  # ---------------------------------------------------------------------------
  # Training helpers
  # ---------------------------------------------------------------------------

  def _add_transition(self, state, probs):
    """Adds a transition to the supervised reservoir buffer."""
    legal_actions = state.legal_actions()
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0
    transition = Transition(
        info_state=state.info_state_tensor()[:],
        action_probs=probs,
        legal_actions_mask=legal_actions_mask,
    )
    self._reservoir_buffers[state.current_player()].add(transition)

  def _loss_avg(self, logits, target_action_probs):
    """Cross-entropy loss between target action probs and network logits."""
    log_probs = F.log_softmax(logits, dim=1)
    loss = -torch.sum(target_action_probs * log_probs) / logits.shape[0]
    return loss

  def _learn(self, player_id):
    """Samples from reservoir buffer, performs a backward pass and
    returns the loss together with the gradients (no optimiser step)."""
    # Exit early if there is not enough data to learn from.
    if len(self._reservoir_buffers[player_id]) < max(
        self._batch_size, self._min_buffer_size_to_learn
    ):
      return None

    # Sample a batch from the reservoir buffer.
    transitions = self._reservoir_buffers[player_id].sample(self._batch_size)

    # Convert to tensors (NumPy first for speed, then torch).
    info_states_np = np.asarray([t.info_state for t in transitions], dtype=np.float32)
    action_probs_np = np.asarray([t.action_probs for t in transitions], dtype=np.float32)
    info_states = torch.from_numpy(info_states_np)
    action_probs = torch.from_numpy(action_probs_np)

    # Forward pass & loss computation.
    logits = self._avg_networks[player_id](info_states)
    loss_val = self._loss_avg(logits, action_probs)

    # Backward pass – we compute gradients but DO NOT apply an optimiser step.
    self._optimizers[player_id].zero_grad()
    loss_val.backward()

    # Collect gradients to return.
    gradients = {
        name: param.grad.detach().clone()
        for name, param in self._avg_networks[player_id].named_parameters()
        if param.grad is not None
    }

    # Return both the scalar loss value and the gradients.
    return gradients, loss_val.item()

  # -------------------------------------------------
  # Checkpointing utilities – Not yet implemented 
  # -------------------------------------------------

  def _full_checkpoint_name(self, checkpoint_dir, name):
    filename = "_".join([name, f"pid{self.player_id}"])
    return os.path.join(checkpoint_dir, filename)

  def _latest_checkpoint_filename(self, name):
    filename = "_".join([name, f"pid{self.player_id}"])
    return f"{filename}_latest"

  def save(self, checkpoint_dir):  # pylint: disable=unused-argument
    """Saves agent parameters. (Not yet implemented)."""
    raise NotImplementedError

  def has_checkpoint(self, checkpoint_dir):
    for name, _ in self._savers:
      path = self._full_checkpoint_name(checkpoint_dir, name)
      if os.path.exists(path):
        return True
    return False

  def restore(self, checkpoint_dir):  # pylint: disable=unused-argument
    """Restores agent parameters. (Not yet implemented)."""
    raise NotImplementedError 