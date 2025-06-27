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


class NFSP(rl_agent.AbstractAgent):
  """NFSP Agent implementation in PyTorch."""

  def __init__(
      self,
      game,
      num_players: int,
      num_actions: int,
      replay_buffer_capacity: int,
      reservoir_buffer_capacity: int,
      anticipatory_param: float,
      avg_net_optimizers: List[torch.optim.Optimizer],
      avg_networks: List[torch.nn.Module],
      q_net_optimizers: List[torch.optim.Optimizer],
      q_networks: List[torch.nn.Module],
      update_target_network_every: int = 1000,
      epsilon_start: float = 0.08,
      epsilon_end: float = 0.001,
      epsilon_decay_duration: int = int(3e6),
      batch_size: int = 512,
      min_buffer_size_to_learn: int = 1000,
      learn_every: int = 128,
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
        replay_buffer_capacity=replay_buffer_capacity,
        batch_size=batch_size,
        update_target_network_every=update_target_network_every,
        min_buffer_size_to_learn=min_buffer_size_to_learn,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_duration=epsilon_decay_duration,
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
  
  def traverse_game(self, episode_num):
    """Traverse a complete game episode."""
    state = self._game.new_initial_state()
    while not state.is_terminal():
      if state.is_chance_node():
        legal_actions = state.legal_actions()
        action = np.random.choice(legal_actions)
        state.apply_action(action)
      else:
        action, _ = self.step(state, state.current_player())
        state.apply_action(action)
    
    sl_gradients = [None for _ in range(self._num_players)]
    br_gradients = [None for _ in range(self._num_players)]
    # final step for both players
    for player_id in range(self._num_players):
      self.step(state, player_id)
    if episode_num % self._learn_every == 0:
      for player_id in range(self._num_players):
        sl_gradient, sl_loss, br_gradient, rl_loss = self.learn_br_rl(player_id)
        self._last_sl_loss_values[player_id] = sl_loss
        self._last_rl_loss_values[player_id] = rl_loss
        sl_gradients[player_id] = sl_gradient
        br_gradients[player_id] = br_gradient
      
    self._sample_episode_policy()
    self._prev_state = [None for _ in range(self._num_players)]
    self._prev_action = [None for _ in range(self._num_players)]
    return sl_gradients, br_gradients
    
    

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
    probs = None
    if is_evaluation:
      mode = MODE.average_policy
    else:
      mode = self._modes[player_id]
    if mode == MODE.best_response:
      action, probs = self._rl_agents[player_id].step(state, is_evaluation)
      if not is_evaluation and not state.is_terminal():
        self._add_transition(state, probs, player_id)
  
    elif mode == MODE.average_policy:
      if not state.is_terminal():
        # Fetch the perspective of the *acting* player, not the default
        # current_player(), otherwise both agents would sometimes receive the
        # wrong private card information in Kuhn Poker and similar games.
        info_state = state.information_state_tensor(player_id)
        legal_actions = state.legal_actions(player_id)
        action, probs = self._act(info_state, legal_actions, player_id)

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
    
    if not is_evaluation:
      self._step_counters[player_id] += 1
      if not state.is_terminal():
        # Store a CLONE of the current state so that subsequent in-place
        # modifications (state.apply_action) do not mutate the experience we
        # are about to record in the replay buffer. Using the un-cloned state
        # would result in transitions whose "previous" and "next" states are
        # actually identical, which severely hurts learning.
        self._prev_state[player_id] = state.clone()
        self._prev_action[player_id] = action
      else:
        # Episode finished – clear stored previous references for this player
        self._prev_state[player_id] = None
        self._prev_action[player_id] = None
    return action, probs
  
  def learn_br_rl(self, player_id):    
    """Learn both the supervised (average policy) and reinforcement learning
    (best-response) networks for `player_id` and keep track of the losses.

    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        The supervised-learning loss and the RL loss obtained in this update.
    """
    # Supervised-learning (SL) update on the reservoir buffer.
    sl_gradients, sl_loss = self._learn(player_id)

    # Reinforcement-learning (RL) update on the replay buffer of the DQN.
    br_gradients, br_loss = self._rl_agents[player_id].learn()

    # Record the most recent loss values so they can be inspected/logged.
    self._last_sl_loss_values[player_id] = sl_loss
    self._last_rl_loss_values[player_id] = br_loss

    return sl_gradients, sl_loss, br_gradients, br_loss
  
  
  # ---------------------------------------------------------------------------
  # Training helpers
  #---------------------------------------------------------------------------

  def _add_transition(self, state, probs, player_id):
    """Adds a transition to the supervised reservoir buffer."""
    legal_actions = state.legal_actions(player_id)
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0
    transition = Transition(
        info_state=state.information_state_tensor(player_id),
        action_probs=probs,
        legal_actions_mask=legal_actions_mask,
    )
    self._reservoir_buffers[player_id].add(transition)

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

    # Collect the gradients for each parameter of the player's average-policy
    # network.  We copy the gradients to CPU and detach them so that they can
    # be sent through Ray (or returned locally) without holding any graph
    # references.  This dictates the structure expected by the Learner:
    # {param_name (str) : grad_tensor (torch.Tensor)}
    gradients = {}
    for name, param in self._avg_networks[player_id].named_parameters():
      if param.grad is not None:
        # Clone & detach → move to CPU so it is serialisable.
        gradients[name] = param.grad.detach().cpu().clone()
    # Store the loss for potential debugging / logging.
    self._last_sl_loss_values[player_id] = loss_val.item()
    # Return both loss and gradients.  The calling code only needs the
    # gradients, but returning the loss can be handy and is inexpensive.
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