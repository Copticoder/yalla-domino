import torch
import dataclasses
from typing import Tuple, Any
from tensordict import TensorDict
def v_trace(
    v: torch.Tensor,
    valid: torch.Tensor,
    player_id: torch.Tensor,
    acting_policy: torch.Tensor,
    merged_policy: torch.Tensor,
    merged_log_policy: torch.Tensor,
    player_others: torch.Tensor,
    actions_oh: torch.Tensor,
    reward: torch.Tensor,
    player: int,
    # Scalars below.
    eta: float,
    lambda_: float,
    c: float,
    rho: float,
) -> Tuple[Any, Any, Any]:
  """Custom VTrace for trajectories with a mix of different player steps."""
  gamma = 1.0

  has_played = player_k_has_played(valid, player_id, player)

  policy_ratio = _policy_ratio(merged_policy, acting_policy, actions_oh, valid)
  inv_mu = _policy_ratio(
      torch.ones_like(merged_policy), acting_policy, actions_oh, valid)

  eta_reg_entropy = (-eta *
                     torch.sum(merged_policy * merged_log_policy, axis=-1) *
                     torch.squeeze(player_others, axis=-1))
  eta_log_policy = -eta * merged_log_policy * player_others

  v_trace_carry = TensorDict(
    reward=torch.zeros_like(reward[-1]),
    reward_uncorrected=torch.zeros_like(reward[-1]),
    next_value=torch.zeros_like(v[-1]),
    next_v_target=torch.zeros_like(v[-1]),
    importance_sampling=torch.ones_like(policy_ratio[-1])
  )
  v_target = torch.zeros_like(v)
  learning_output = torch.zeros_like(v)
  for t in range(len(v)-2, -1, -1):
    cs = has_played*torch.minimum(1, policy_ratio[t+1] * v_trace_carry.importance_sampling[t+1])
    rho = torch.minimum(1, policy_ratio[t+1] * v_trace_carry.importance_sampling[t+1])
    r_uncorrected = reward[:,t,:] + gamma * v_trace_carry.reward_uncorrected[t+1]
    reward_corrected = (1-has_played[:,t,:])*(reward[:,t,:] + policy_ratio[:,t,:]*v_trace_carry.reward[t+1]) + (has_played[:,t,:])*()
    v_target[t] = rho*()
def _policy_ratio(pi: torch.Tensor, mu: torch.Tensor, actions_oh: torch.Tensor,
                  valid: torch.Tensor) -> torch.Tensor:
  """Returns a ratio of policy pi/mu when selecting action a.

  By convention, this ratio is 1 on non valid states
  Args:
    pi: the policy of shape [..., A].
    mu: the sampling policy of shape [..., A].
    actions_oh: a one-hot encoding of the current actions of shape [..., A].
    valid: boolean tensor indicating valid states of shape [...,1].

  Returns:
    pi/mu on valid states and 1 otherwise. The shape is the same
    as pi, mu or actions_oh but without the last dimension A.
  """
  
  assert pi.shape == mu.shape == actions_oh.shape, "pi, mu, and actions_oh must have the same shape"
  assert valid.shape == actions_oh.shape[:-1], "valid must have the same shape as actions_oh except the last dimension"
  assert valid.dtype == torch.bool, "valid must be a boolean tensor"

  def _select_action_prob(pi):
    return (torch.sum(actions_oh * pi, axis=-1, keepdims=False) * valid +
            ~valid)

  pi_actions_prob = _select_action_prob(pi)
  mu_actions_prob = _select_action_prob(mu)
  return pi_actions_prob / mu_actions_prob 

def _where(pred: chex.Array, true_data: chex.ArrayTree,
           false_data: chex.ArrayTree) -> chex.ArrayTree:
  """Similar to jax.where but treats `pred` as a broadcastable prefix."""

  def _where_one(t, f):
    chex.assert_equal_rank((t, f))
    # Expand the dimensions of pred if true_data and false_data are higher rank.
    p = jnp.reshape(pred, pred.shape + (1,) * (len(t.shape) - len(pred.shape)))
    return jnp.where(p, t, f)

  return tree.tree_map(_where_one, true_data, false_data)

def player_k_has_played(valid: torch.Tensor, player_id: torch.Tensor,
                player: int) -> torch.Tensor:
    """Compute a mask where player k has played in the sequence. Useful for accurately calculating the value function when it's player k's turn."""
    # assert that valid and player_id are the same shape
    assert valid.shape == player_id.shape, "valid and player_id must have the same shape"
    # assert that valid is a boolean tensor
    if valid.dtype != torch.bool:
        valid = valid.bool()
    # create a new tensor player_k_has_played with the same shape as valid 
    player_k_has_played = torch.zeros_like(valid)
    # set player_k_has_played[t] to 1 if valid[t] and player_id[t] == player
    player_k_has_played[valid & (player_id == player)] = 1
    return player_k_has_played

def player_others(player_ids: torch.Tensor, valid: torch.Tensor,
                   player: int) -> torch.Tensor:
  """A vector of 1 for the current player and -1 for others.

  Args:
    player_ids: Tensor [...] containing player ids (0 <= player_id < N).
    valid: Tensor [...] containing whether these states are valid.
    player: The player id as int.

  Returns:
    player_other: is 1 for the current player and -1 for others [..., 1].
  """
  assert player_ids.shape == valid.shape, "player_ids and valid must have the same shape"
  current_player_tensor = (player_ids == player) 

  res = 2 * current_player_tensor - 1
  res = res * valid
  return res.unsqueeze(-1)