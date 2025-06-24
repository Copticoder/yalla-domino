import torch
from typing import Tuple, Any, Sequence
from tensordict import TensorDict
from dataclasses import dataclass

def get_loss_v(v_list: Sequence[torch.Tensor],
               v_target_list: Sequence[torch.Tensor],
               mask_list: Sequence[torch.Tensor]) -> torch.Tensor:
  """Define the loss function for the critic."""
  assert all(v.shape == v_target.shape for v, v_target in zip(v_list, v_target_list))
  # v_list and v_target_list come with a degenerate trailing dimension,
  # which mask_list tensors do not have.
  assert mask_list[0].shape == v_list[0].shape[:-1]  # mask has one less dim
  loss_v_list = []
  for (v_n, v_target, mask) in zip(v_list, v_target_list, mask_list):
    assert v_n.shape[0] == v_target.shape[0]

    loss_v = mask.unsqueeze(-1) * (v_n - v_target.detach())**2  # Add dim for broadcasting
    normalization = torch.sum(mask)
    loss_v = torch.sum(loss_v) / (normalization + (normalization == 0.0))

    loss_v_list.append(loss_v)
  return torch.sum(torch.stack(loss_v_list))


def apply_force_with_threshold(decision_outputs: torch.Tensor, force: torch.Tensor,
                               threshold: float,
                               threshold_center: torch.Tensor) -> torch.Tensor:
  """Apply the force with below a given threshold."""
  assert decision_outputs.shape == force.shape == threshold_center.shape
  can_decrease = decision_outputs - threshold_center > -threshold
  can_increase = decision_outputs - threshold_center < threshold
  force_negative = torch.minimum(force, torch.zeros_like(force))
  force_positive = torch.maximum(force, torch.zeros_like(force))
  clipped_force = can_decrease * force_negative + can_increase * force_positive
  return decision_outputs * clipped_force.detach()


def renormalize(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  """The `normalization` is the number of steps over which loss is computed."""
  assert loss.shape == mask.shape
  loss = torch.sum(loss * mask)
  normalization = torch.sum(mask)
  return loss / (normalization + (normalization == 0.0))


def get_loss_nerd(logit_list: Sequence[torch.Tensor],
                  policy_list: Sequence[torch.Tensor],
                  q_vr_list: Sequence[torch.Tensor],
                  valid: torch.Tensor,
                  player_ids: torch.Tensor,  # Single tensor, not list
                  legal_actions: torch.Tensor,
                  importance_sampling_correction: Sequence[torch.Tensor],
                  clip: float = 100,
                  threshold: float = 2) -> torch.Tensor:
  """Define the nerd loss."""
  assert isinstance(importance_sampling_correction, list)
  loss_pi_list = []
  num_valid_actions = torch.sum(legal_actions, dim=-1, keepdim=True)
  for k, (logit_pi, pi, q_vr, is_c) in enumerate(
      zip(logit_list, policy_list, q_vr_list, importance_sampling_correction)):
    assert logit_pi.shape[0] == q_vr.shape[0]
    # loss policy
    adv_pi = q_vr - torch.sum(pi * q_vr, dim=-1, keepdim=True)
    adv_pi = is_c * adv_pi  # importance sampling correction
    adv_pi = torch.clamp(adv_pi, min=-clip, max=clip)
    adv_pi = adv_pi.detach()

    valid_logit_sum = torch.sum(logit_pi * legal_actions, dim=-1, keepdim=True)
    mean_logit = valid_logit_sum / num_valid_actions

    # Subtract only the mean of the valid logits
    logits = logit_pi - mean_logit

    threshold_center = torch.zeros_like(logits)

    nerd_loss = torch.sum(
        legal_actions *
        apply_force_with_threshold(logits, adv_pi, threshold, threshold_center),
        dim=-1)
    nerd_loss = -renormalize(nerd_loss, valid * (player_ids == k))
    loss_pi_list.append(nerd_loss)
  return torch.sum(torch.stack(loss_pi_list))

# Test the loss function
if __name__ == "__main__":
  batch_size, seq_len, num_actions = 10, 8, 5
  
  logit_list = [torch.randn(batch_size, seq_len, num_actions), torch.randn(batch_size, seq_len, num_actions)]
  policy_list = [torch.softmax(torch.randn(batch_size, seq_len, num_actions), dim=-1), 
                 torch.softmax(torch.randn(batch_size, seq_len, num_actions), dim=-1)]
  q_vr_list = [torch.randn(batch_size, seq_len, num_actions), torch.randn(batch_size, seq_len, num_actions)]
  valid = torch.ones(batch_size, seq_len, dtype=torch.bool)  # Boolean mask
  player_ids = torch.randint(0, 2, (batch_size, seq_len))  # Single tensor with player IDs
  legal_actions = torch.ones(batch_size, seq_len, num_actions, dtype=torch.bool)  # Boolean mask
  importance_sampling_correction = [torch.ones(batch_size, seq_len, num_actions), 
                                   torch.ones(batch_size, seq_len, num_actions)]
  
  loss = get_loss_nerd(logit_list, policy_list, q_vr_list, valid, player_ids, legal_actions, importance_sampling_correction)
  print(f"NERD loss: {loss.item():.4f}")