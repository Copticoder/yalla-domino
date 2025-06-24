
import torch
from typing import Tuple

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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

  def v_trace_single_sequence(v_seq, has_played_seq, reward_seq, inv_mu_seq, eta_reg_entropy_seq, eta_log_policy_seq, actions_oh_seq, policy_ratio_seq):
    """Process a single sequence (one batch element) for v-trace."""
    T = len(v_seq) - 1  # v has shape [T+1], others have shape [T]
    
    # Initialize carry state for the backward sweep
    reward_corrected = torch.zeros_like(reward_seq[-1])
    next_value = torch.zeros_like(v_seq[-1]).squeeze()  # Remove extra dims
    next_v_target = torch.zeros_like(v_seq[-1]).squeeze()
    importance_sampling = torch.ones_like(policy_ratio_seq[-1])  # scalar
    
    # Pre-allocate output tensors
    v_targets = torch.zeros_like(v_seq).squeeze()  # [T]
    learning_outputs = torch.zeros_like(actions_oh_seq)  # [T, A]
    
    for t in reversed(range(T)):
      # Compute importance sampling ratios
      rho_t = torch.minimum(torch.tensor(rho), policy_ratio_seq[t] * importance_sampling)
      c_t = torch.minimum(torch.tensor(c), policy_ratio_seq[t] * importance_sampling)
      
      # Use torch.where instead of if statements for vmap compatibility
      has_played_t = has_played_seq[t]
      
      # Compute values for both cases (player's turn and opponent's turn)
      # Player's turn computations
      bootstrap_v_target_t = rho_t * (reward_seq[t] + eta_reg_entropy_seq[t]*reward_corrected + gamma * next_value - v_seq[t].squeeze())
      v_target_player = v_seq[t].squeeze() + bootstrap_v_target_t + lambda_ * c_t * gamma * (next_v_target - next_value)
      
      learning_output_player = (eta_log_policy_seq[t] + 
                               actions_oh_seq[t] * inv_mu_seq[t].unsqueeze(-1) * 
                               (reward_seq[t] + eta_reg_entropy_seq[t] + gamma * importance_sampling * (reward_corrected + next_v_target) - v_seq[t].squeeze()))
      
      reward_corrected_player = torch.zeros_like(reward_seq[t])
      importance_sampling_player = torch.ones_like(importance_sampling)
      next_value_player = v_target_player
      next_v_target_player = v_target_player
      
      # Opponent's turn computations  
      v_target_opponent = next_v_target
      learning_output_opponent = torch.zeros_like(actions_oh_seq[t])
      reward_corrected_opponent = eta_reg_entropy_seq[t] + c_t * reward_corrected
      importance_sampling_opponent = c_t * importance_sampling
      next_value_opponent = gamma * v_seq[t+1].squeeze()
      next_v_target_opponent = gamma * v_target_opponent
      
      # Select based on has_played using torch.where
      v_target = torch.where(has_played_t, v_target_player, v_target_opponent)
      learning_output = torch.where(has_played_t.unsqueeze(-1), learning_output_player, learning_output_opponent)
      reward_corrected = torch.where(has_played_t, reward_corrected_player, reward_corrected_opponent)
      importance_sampling = torch.where(has_played_t, importance_sampling_player, importance_sampling_opponent)
      next_value = torch.where(has_played_t, next_value_player, next_value_opponent)
      next_v_target = torch.where(has_played_t, next_v_target_player, next_v_target_opponent)
      
      # Store results
      v_targets[t] = v_target
      learning_outputs[t] = learning_output
    
    return v_targets, learning_outputs
  
  # Apply vmap over the batch dimension
  v_targets, learning_outputs = torch.vmap(v_trace_single_sequence, in_dims=(0, 0, 0, 0, 0, 0, 0, 0))(
      v, has_played, reward, inv_mu, eta_reg_entropy, eta_log_policy, actions_oh, policy_ratio
  )
  
  return v_targets, has_played, learning_outputs

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



# --------------------------------------------------------------------
# print(v.shape, valid.shape,player_id.shape, merged_policy.shape, player_others.shape, actions_oh.shape, reward.shape)
# torch.Size([256, 10, 1, 1]) torch.Size([256, 10]) torch.Size([256, 10]) torch.Size([256, 10, 3]) torch.Size([256, 10, 1]) torch.Size([256, 10, 3]) torch.Size([256, 10])

v = torch.randn(256, 10, 1, 1)
valid = torch.rand(256, 10) < 0.5
player_id = torch.rand(256, 10) < 0.5
merged_policy = torch.randn(256, 10, 3)
merged_log_policy = torch.randn(256, 10, 3)
acting_policy = torch.randn(256, 10, 3)
actions_oh = torch.randn(256, 10, 3)
reward = torch.randn(256, 10)
player = 0
eta = 0.1
lambda_ = 0.9
c = 0.1
rho = 0.1
v_trace(v, valid, player_id, acting_policy, merged_policy, merged_log_policy, player_others(player_id, valid, player), actions_oh, reward, player, eta, lambda_, c, rho)
# --------------------------------------------------------------------
# Test the v-trace function
print("Testing v_trace function...")

v = torch.randn(256, 10, 1, 1)
valid = torch.rand(256, 10) < 0.8  # 80% valid states
player_id = torch.randint(0, 2, (256, 10))  # Binary player IDs (0 or 1)
merged_policy = torch.softmax(torch.randn(256, 10, 3), dim=-1)  # Valid probabilities
merged_log_policy = torch.log(merged_policy + 1e-8)
acting_policy = torch.softmax(torch.randn(256, 10, 3), dim=-1)  # Valid probabilities
actions_oh = torch.zeros(256, 10, 3)
# Create proper one-hot actions
for i in range(256):
    for j in range(10):
        action_idx = torch.randint(0, 3, (1,))
        actions_oh[i, j, action_idx] = 1.0

reward = torch.randn(256, 10)
player = 0
eta = 0.1
lambda_ = 0.9
c = 0.1
rho = 0.1

print(f"Input shapes:")
print(f"v: {v.shape}")
print(f"valid: {valid.shape}")
print(f"player_id: {player_id.shape}")
print(f"merged_policy: {merged_policy.shape}")
print(f"actions_oh: {actions_oh.shape}")
print(f"reward: {reward.shape}")

v_targets, has_played, learning_outputs = v_trace(
    v, valid, player_id, acting_policy, merged_policy, merged_log_policy, 
    player_others(player_id, valid, player), actions_oh, reward, player, 
    eta, lambda_, c, rho
)

print(f"\nOutput shapes:")
print(f"v_targets: {v_targets.shape}")
print(f"has_played: {has_played.shape}")
print(f"learning_outputs: {learning_outputs.shape}")

print(f"\nOutput value ranges:")
print(f"v_targets range: [{v_targets.min().item():.3f}, {v_targets.max().item():.3f}]")
print(f"learning_outputs range: [{learning_outputs.min().item():.3f}, {learning_outputs.max().item():.3f}]")
print(f"has_played sum: {has_played.sum().item()} (out of {has_played.numel()} total)")

print("✓ v_trace function test completed successfully!")


