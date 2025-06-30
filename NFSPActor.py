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
import enum
import os
from typing import List
from MLPs import BR_MLP, AVG_MLP

import numpy as np
import torch
import torch.nn.functional as F
import ray
from open_spiel.python import rl_agent
from DQN import DQN
from open_spiel.python.utils.reservoir_buffer import ReservoirBuffer
Transition = collections.namedtuple(
    "Transition", "info_state action_probs legal_actions_mask")

MODE = enum.Enum("mode", "best_response average_policy")

@ray.remote(num_cpus=1, namespace="nfsp")
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
      learning_rate: float,
      state_representation_size: int,
      hidden_layers_sizes: List[int],
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
    
    # Create networks and optimizers locally in each actor
    self._q_networks = [BR_MLP(state_representation_size, hidden_layers_sizes, num_actions) for _ in range(num_players)]
    self._avg_networks = [AVG_MLP(state_representation_size, hidden_layers_sizes, num_actions) for _ in range(num_players)]
    self._q_net_optimizers = [torch.optim.SGD(q_network.parameters(), lr=learning_rate) for q_network in self._q_networks]
    self._avg_net_optimizers = [torch.optim.SGD(avg_network.parameters(), lr=learning_rate) for avg_network in self._avg_networks]
    
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
        optimizer=self._q_net_optimizers[player_id],
        q_network=self._q_networks[player_id],
        loss_str="mse",
    ) for player_id in range(num_players)]

    # Keep track of the last training loss achieved in an update step.
    self._last_rl_loss_values = [None for _ in range(num_players)]
    self._last_sl_loss_values = [None for _ in range(num_players)]

    # Optimizer for supervised learning (SL) network.
    self._optimizers = self._avg_net_optimizers

    self._savers = [
        ("q_network", self._rl_agents),
        ("avg_network", self._avg_networks),
    ]
  # ---------------------------------------------------------------------------
  # Federated Learning Methods
  # ---------------------------------------------------------------------------
  
  def update_networks(self, q_network_params, avg_network_params):
    """Update the networks with new parameters from the learner.
    
    Args:
      q_network_params: List of state dictionaries for Q-networks
      avg_network_params: List of state dictionaries for average policy networks
    """
    for player_id in range(self._num_players):
      # Update Q-network parameters
      self._q_networks[player_id].load_state_dict(q_network_params[player_id])
      self._rl_agents[player_id]._q_network.load_state_dict(q_network_params[player_id])
      
      # Update average policy network parameters
      self._avg_networks[player_id].load_state_dict(avg_network_params[player_id])
      
      # Update target Q-network parameters for DQN
      self._rl_agents[player_id]._target_q_network.load_state_dict(q_network_params[player_id])
  
  def compute_gradients(self, player_id):
    """Compute gradients for both average policy and Q-networks for a specific player.
    
    Args:
      player_id: Player ID to compute gradients for
      
    Returns:
      Tuple of (avg_policy_gradients, q_network_gradients, sl_loss, rl_loss)
    """
    # Compute supervised learning (average policy) gradients
    avg_gradients, sl_loss = self._learn(player_id, return_gradients=True)
    
    # Compute reinforcement learning (Q-network) gradients  
    q_gradients, rl_loss = self._rl_agents[player_id].learn(return_gradients=True)
    # save losses 
    self._last_rl_loss_values[player_id] = rl_loss
    self._last_sl_loss_values[player_id] = sl_loss
    return avg_gradients, q_gradients, sl_loss, rl_loss
    
  # ---------------------------------------------------------------------------
  # Acting
  # ---------------------------------------------------------------------------
  
  def traverse_game(self, itr, modes):
    """Traverse a complete game episode."""
    state = self._game.new_initial_state()
    while not state.is_terminal():
      if state.is_chance_node():
        legal_actions = state.legal_actions()
        action = np.random.choice(legal_actions)
        state.apply_action(action)
      else:
        action, _ = self.step(state, state.current_player(), modes[state.current_player()])
        state.apply_action(action)
    # final step for both players
    for player_id in range(self._num_players):
      self.step(state, player_id, modes[player_id])
    
    # Note: Removed local learning here - gradients will be computed and aggregated by learner
    self._prev_state = [None for _ in range(self._num_players)]
    self._prev_action = [None for _ in range(self._num_players)]

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
  def step(self, state, player_id, mode, is_evaluation: bool = False):
    """Returns the action to be taken and updates the networks if needed."""
    action = None
    probs = None
    if is_evaluation:
      mode = MODE.average_policy
      
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
    sl_loss = self._learn(player_id)

    # Reinforcement-learning (RL) update on the replay buffer of the DQN.
    rl_loss = self._rl_agents[player_id].learn()

    # Record the most recent loss values so they can be inspected/logged.
    self._last_sl_loss_values[player_id] = sl_loss
    self._last_rl_loss_values[player_id] = rl_loss

    return sl_loss, rl_loss
  
  
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

  def _learn(self, player_id, return_gradients=False):
    """Samples from reservoir buffer, performs a backward pass and
    optionally returns gradients or applies optimizer step."""
    # Exit early if there is not enough data to learn from.
    if len(self._reservoir_buffers[player_id]) < max(
        self._batch_size, self._min_buffer_size_to_learn
    ):
      if return_gradients:
        return None, None
      else:
        return None

    # Sample a batch from the reservoir buffer.
    transitions = self._reservoir_buffers[player_id].sample(self._batch_size)

    # Convert to tensors (NumPy first for speed, then torch).
    info_states_np = np.asarray([t.info_state for t in transitions], dtype=np.float32)
    action_probs_np = np.asarray([t.action_probs for t in transitions], dtype=np.float32)
    info_states = torch.from_numpy(info_states_np)
    action_probs = torch.from_numpy(action_probs_np)

    # Debug: Get network params before update
    if hasattr(self, '_debug_param_count'):
        self._debug_param_count += 1
    else:
        self._debug_param_count = 1
    
    param_before = None
    if self._debug_param_count % 1000 == 0:  # Check every 1000 updates
        param_before = list(self._avg_networks[player_id].parameters())[0].clone()

    # Forward pass & loss computation.
    logits = self._avg_networks[player_id](info_states)
    loss_val = self._loss_avg(logits, action_probs)

    # Backward pass – compute gradients
    self._optimizers[player_id].zero_grad()
    loss_val.backward()
    
    if return_gradients:
        # Extract gradients
        gradients = {}
        for name, param in self._avg_networks[player_id].named_parameters():
          if param.grad is not None:
            gradients[name] = param.grad.clone()
        return gradients, loss_val.item()
    else:
        # Apply optimizer step
        self._optimizers[player_id].step()
        
        return loss_val.item()

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