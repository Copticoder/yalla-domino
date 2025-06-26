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

@ray.remote(num_cpus=1, namespace="nfsp")
class NFSP(rl_agent.AbstractAgent):
  """NFSP Agent implementation in PyTorch."""

  def __init__(
      self,
      game,
      num_players: int,
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
  ) -> None:
    self._game = game
    self._num_actions = num_actions
    self._batch_size = batch_size
    self._learn_every = learn_every
    self._anticipatory_param = anticipatory_param
    self._min_buffer_size_to_learn = min_buffer_size_to_learn
    self._num_players = num_players
    self._reservoir_buffers = [ReservoirBuffer(reservoir_buffer_capacity) for _ in range(num_players)]
    self._prev_state = [None for _ in range(num_players)]
    self._prev_action = [None for _ in range(num_players)]

    # Step counter to keep track of learning.
    self._step_counters = [0 for _ in range(num_players)]
    self._rl_agents = [DQN(
        player_id=player_id,
        num_actions=num_actions,
        replay_buffer_capacity=reservoir_buffer_capacity,
        batch_size=batch_size,
        update_target_network_every=128,
        min_buffer_size_to_learn=min_buffer_size_to_learn,
        epsilon_start=0.08,
        epsilon_end=0.001,
        epsilon_decay_duration=1000000,
        optimizer=q_net_optimizers[player_id],
        q_network=q_networks[player_id],
        loss_str="mse",
    ) for player_id in range(num_players)]

    # Keep track of the last training loss achieved in an update step.
    self._last_rl_loss_values = [None for _ in range(num_players)]
    self._last_sl_loss_values = [None for _ in range(num_players)]

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
    """Update the networks and optimizers with new parameters.
    
    Args:
      q_networks: List of state dictionaries for Q-networks
      avg_networks: List of state dictionaries for average policy networks  
      q_net_optimizers: List of state dictionaries for Q-network optimizers
      avg_net_optimizers: List of state dictionaries for average policy optimizers
    """
    for i in range(self._num_players):
      # Load Q-network parameters
      self._rl_agents[i]._q_network.load_state_dict(q_networks[i])
      # Load average policy network parameters
      self._avg_networks[i].load_state_dict(avg_networks[i])
      # Load optimizer states
      self._rl_agents[i]._optimizer.load_state_dict(q_net_optimizers[i])
      self._optimizers[i].load_state_dict(avg_net_optimizers[i])
  
  # ---------------------------------------------------------------------------
  # Acting
  # ---------------------------------------------------------------------------
  
  def traverse_game(self):
    """Traverse a complete game episode."""
    state = self._game.new_initial_state()
    while not state.is_terminal():
      if state.is_chance_node():
        legal_actions = state.legal_actions()
        action = np.random.choice(legal_actions)
        state = state.child(action)
      else:
        action = self.step(state, state.current_player())
        state = state.child(action)
    # final step for both players
    for player_id in range(self._num_players):
      self.step(state, player_id)
    self._prev_state = [None for _ in range(self._num_players)]
    self._prev_action = [None for _ in range(self._num_players)]
    self._sample_episode_policy()
    
    

  def _sample_episode_policy(self):
    # Sample an episode policy *independently for each player* so that
    # best-response / average-policy episodes are not perfectly
    # synchronised across all players.  This matches the design of the
    # reference implementation where each NFSP agent (one per player)
    # samples its own mode.
    self._modes = []
    for _ in range(self._num_players):
      if np.random.rand() < self._anticipatory_param:
        self._modes.append(MODE.best_response)
      else:
        self._modes.append(MODE.average_policy)

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
    # Expose the list of per-player modes for inspection.
    return self._modes

  # Adding a helper method for Ray remote access to loss since @property cannot be
  # accessed with `.remote()` syntax from an actor handle. This ensures the
  # ParameterServer can query the current loss safely.
  def get_loss(self, player_id):
    """Return current supervised and reinforcement learning losses."""
    return self._last_sl_loss_values[player_id], self._last_rl_loss_values[player_id]

  # ---------------------------------------------------------------------------
  # RL-Agent compatible interface
  # ---------------------------------------------------------------------------
  def step(self, state, player_id, is_evaluation: bool = False):
    """Returns the action to be taken and updates the networks if needed."""
    action = None
    mode = self._modes[player_id]
    if mode == MODE.best_response:
      action, probs = self._rl_agents[player_id].step(state, is_evaluation)
      if not is_evaluation and not state.is_terminal():
        self._add_transition(state, probs)
  
    elif mode == MODE.average_policy:
      if not state.is_terminal():
        info_state = state.information_state_tensor()
        legal_actions = state.legal_actions()
        action, _ = self._act(info_state, legal_actions, player_id)

      # Feed the (s,a,s') transition to the RL agent using the *previous*
      # state/action stored for this player, exactly as in the reference
      # OpenSpiel NFSP implementation.
      if (
          not is_evaluation
          and self._prev_state[player_id] is not None
      ):
        self._rl_agents[player_id].add_transition(
            self._prev_state[player_id],
            self._prev_action[player_id],
            state,
        )

    else:
      raise ValueError(f"Invalid mode ({mode})")
    
    if not state.is_terminal():
      self._prev_state[player_id] = state
      self._prev_action[player_id] = action
    else:
      # Episode finished – clear stored previous references for this player
      self._prev_state[player_id] = None
      self._prev_action[player_id] = None
    return action
  
  def learn_br_rl(self, player_id):    
    """Learn and return gradients for both average and Q networks."""
    try:
      gradients_sl, self._last_sl_loss_values[player_id] = self._learn(player_id)
      gradients_rl, self._last_rl_loss_values[player_id] = self._rl_agents[player_id].learn()
      
      # Return empty gradients if learning didn't happen due to insufficient data
      if gradients_sl is None:
        gradients_sl = {}
      if gradients_rl is None:
        gradients_rl = {}
        
      return gradients_sl, gradients_rl
      
    except Exception as e:
      print(f"Error in learn_br_rl for player {player_id}: {e}")
      # Return empty gradient dictionaries instead of None
      return {}, {}
  
  # ---------------------------------------------------------------------------
  # Training helpers
  # ---------------------------------------------------------------------------

  def _add_transition(self, state, probs):
    """Adds a transition to the supervised reservoir buffer."""
    legal_actions = state.legal_actions()
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0
    transition = Transition(
        info_state=state.information_state_tensor()[:],
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
      return None, None

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