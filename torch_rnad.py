from typing import Sequence, Tuple
import numpy as np
import torch
import enum
from open_spiel.python import policy
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.optim import Adam, SGD
import dataclasses
from torchrl.data import SliceSampler
from torchrl.data.replay_buffers import TensorDictReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
import torch
import dataclasses
import pyspiel

# Define ActorCriticNetwork at the module level
class ActorCriticNetwork(nn.Module):
  def __init__(self, input_dim, hidden_layers=(128, 128), num_actions=4):
      super(ActorCriticNetwork, self).__init__()
      self.input_dim = input_dim
      self.hidden_layers = hidden_layers
      self.num_actions = num_actions
      # Define the shared MLP layers
      layers = []
      prev_dim = input_dim
      for hidden_dim in hidden_layers:
          layers.append(nn.Linear(prev_dim, hidden_dim))
          layers.append(nn.ReLU())
          prev_dim = hidden_dim
      self.shared_mlp = nn.Sequential(*layers)
      # Define the actor head
      self.actor_head = nn.Linear(prev_dim, num_actions)
      # Define the critic head
      self.critic_head = nn.Linear(prev_dim, 1)
  def forward(self, info_state_tensor, legal_actions_tensor):
      # Convert numpy inputs from EnvStep to torch tensors
      obs_tensor = info_state_tensor.squeeze()
      # legal_actions are used as masks, boolean is appropriate.
      legal_actions_tensor = legal_actions_tensor.squeeze()

      # Pass through the shared MLP
      x = self.shared_mlp(obs_tensor)
      # Actor: output the policy logits
      logits = self.actor_head(x).squeeze()
      # Critic: output the state value
      value = self.critic_head(x).squeeze()

      # Use converted tensors for policy functions
      pi = _legal_policy(logits, legal_actions_tensor)
      log_pi = legal_log_policy(logits, legal_actions_tensor)
      return pi, value, log_pi, logits

class EntropySchedule:
  """An increasing list of steps where the regularisation network is updated.

  Example
    EntropySchedule([3, 5, 10], [2, 4, 1])
    =>   [0, 3, 6, 11, 16, 21, 26, 36]
          | 3 x2 |      5 x4     | 10 x1
  """

  def __init__(self, *, sizes: Sequence[int], repeats: Sequence[int]):
    """Constructs a schedule of entropy iterations.

    Args:
      sizes: the list of iteration sizes.
      repeats: the list, parallel to sizes, with the number of times for each
        size from `sizes` to repeat.
    """
    try:
      if len(repeats) != len(sizes):
        raise ValueError("`repeats` must be parallel to `sizes`.")
      if not sizes:
        raise ValueError("`sizes` and `repeats` must not be empty.")
      if any([(repeat <= 0) for repeat in repeats]):
        raise ValueError("All repeat values must be strictly positive")
      if repeats[-1] != 1:
        raise ValueError("The last value in `repeats` must be equal to 1, "
                         "ince the last iteration size is repeated forever.")
    except ValueError as e:
      raise ValueError(
          f"Entropy iteration schedule: repeats ({repeats}) and sizes"
          f" ({sizes})."
      ) from e

    schedule = [0]
    for size, repeat in zip(sizes, repeats):
      schedule.extend([schedule[-1] + (i + 1) * size for i in range(repeat)])

    self.schedule = np.array(schedule, dtype=np.int32)

  def __call__(self, learner_step: int) -> Tuple[float, bool]:
    """Entropy scheduling parameters for a given `learner_step`.

    Args:
      learner_step: The current learning step.

    Returns:
      alpha: The mixing weight (from [0, 1]) of the previous policy with
        the one before for computing the intrinsic reward.
      update_target_net: A boolean indicator for updating the target network
        with the current network.
    """

    # The complexity below is because at some point we might go past
    # the explicit schedule, and then we'd need to just use the last step
    # in the schedule and apply the logic of
    # ((learner_step - last_step) % last_iteration) == 0)

    # The schedule might look like this:
    # X----X-------X--X--X--X--------X
    # learner_step | might be here ^    |
    # or there     ^                    |
    # or even past the schedule         ^

    # We need to deal with two cases below.
    # Instead of going for the complicated conditional, let's just
    # compute both and then do the A * s + B * (1 - s) with s being a bool
    # selector between A and B.

    # 1. assume learner_step is past the schedule,
    #    ie schedule[-1] <= learner_step.
    last_size = self.schedule[-1] - self.schedule[-2]
    last_start = self.schedule[-1] + (
        learner_step - self.schedule[-1]) // last_size * last_size
    # 2. assume learner_step is within the schedule.
    start = np.amax(self.schedule * (self.schedule <= learner_step))
    finish = np.amin(
        self.schedule * (learner_step < self.schedule),
        initial=self.schedule[-1],
        where=(learner_step < self.schedule))
    size = finish - start

    # Now select between the two.
    beyond = (self.schedule[-1] <= learner_step)  # Are we past the schedule?
    iteration_start = (last_start * beyond + start * (1 - beyond))
    iteration_size = (last_size * beyond + size * (1 - beyond))

    # Ensure the result is a Python bool (not a numpy.bool_) by calling `.item()`.
    update_target_net = np.logical_and(
        learner_step > 0,
        learner_step == (iteration_start + iteration_size - 1),
    ).item()
    alpha = np.minimum(
        (2.0 * (learner_step - iteration_start)) / iteration_size, 1.0)

    return alpha, update_target_net  # pytype: disable=bad-return-type  # jax-types

@dataclasses.dataclass
class FineTuning:
  """Fine tuning options, aka policy post-processing.

  Even when fully trained, the resulting softmax-based policy may put
  a small probability mass on bad actions. This results in an agent
  waiting for the opponent (itself in self-play) to commit an error.

  To address that the policy is post-processed using:
  - thresholding: any action with probability smaller than self.threshold
    is simply removed from the policy.
  - discretization: the probability values are rounded to the closest
    multiple of 1/self.discretization.

  The post-processing is used on the learner, and thus must be jit-friendly.
  """
  # The learner step after which the policy post processing (aka finetuning)
  # will be enabled when learning. A strictly negative value is equivalent
  # to infinity, ie disables finetuning completely.
  from_learner_steps: int = -1
  # All policy probabilities below `threshold` are zeroed out. Thresholding
  # is disabled if this value is non-positive.
  policy_threshold: float = 0.03
  # Rounds the policy probabilities to the "closest"
  # multiple of 1/`self.discretization`.
  # Discretization is disabled for non-positive values.
  policy_discretization: int = 32

  def __call__(self, policy: torch.Tensor, mask: torch.Tensor,
               learner_steps: int) -> torch.Tensor:
    """A configurable fine tuning of a policy."""
    assert policy.shape == mask.shape
    do_finetune = torch.logical_and(torch.tensor(self.from_learner_steps >= 0),
                                   torch.tensor(learner_steps > self.from_learner_steps))

    return torch.where(do_finetune, self.post_process_policy(policy, mask),
                     policy)

  def post_process_policy(
      self,
      policy: torch.Tensor,
      mask: torch.Tensor,
  ) -> torch.Tensor:
    """Unconditionally post process a given masked policy."""
    assert policy.shape == mask.shape
    policy = self._threshold(policy, mask)
    # flatten policy to (B*T,A)
    mu = self._discretize(policy.view(-1,policy.shape[-1]))
    # reshape it back to the original shape (B,T,A)
    policy = mu.view(policy.shape[0],policy.shape[1],-1)
    return policy

  def _threshold(self, policy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Remove from the support the actions 'a' where policy(a) < threshold."""
    assert policy.shape == mask.shape
    if self.policy_threshold <= 0:
      return policy

    mask = mask * (
        # Values over the threshold.
        (policy >= self.policy_threshold) +
        # Degenerate case is when policy is less than threshold *everywhere*.
        # In that case we just keep the policy as-is.
        (torch.max(policy, axis=-1, keepdims=True).values < self.policy_threshold))
    return mask * policy / torch.sum(mask * policy, axis=-1, keepdims=True)

  def _discretize(self, mu: torch.Tensor) -> torch.Tensor:
      """
      Discretize a batch of probability vectors into fixed-resolution distributions.

      Given a batch of "soft" probability vectors `mu` (shape `[B*T, n_actions]`),
      this function allocates exactly `policy_discretization` quanta (by default 32)
      across each vector's entries in descending order of their original values.
      Each entry's allocated quanta is the ceiling of its original weight-times-budget,
      but capped by the remaining budget as we sweep through actions from largest
      to smallest. Any leftover quanta are dumped into the top action. Finally, the
      integer counts are renormalized back into probabilities.

      Parameters
      ----------
      mu : torch.Tensor
          A batch of unbatched probability vectors, shape `(B*T, n_actions)`,
          where each row sums to 1 (or approximately so).

      Returns
      -------
      torch.Tensor
          A tensor of shape `(B*T, n_actions)` where each row is a probability
          vector that (a) sums exactly to 1, and (b) has values that are multiples
          of `1 / policy_discretization`.

      Example
      -------
      >>> mu = torch.tensor([[0.1, 0.2, 0.7],
      ...                    [0.33, 0.33, 0.34]])
      >>> discretize(mu)
      tensor([[0.0000, 0.1875, 0.8125],
              [0.3125, 0.3125, 0.3750]])
      """
      if self.policy_discretization <= 0:
          return mu
      policy_discretization = self.policy_discretization
      n_actions = mu.shape[-1]
      roundup = torch.ceil(mu * policy_discretization)
      result = torch.zeros_like(mu)
      order = torch.argsort(-mu)  # Indices of descending order.
      weight_left = policy_discretization * torch.ones(mu.shape[0])
      for i in range(n_actions):
          next_action = order[:,i]
          x = torch.minimum(roundup[torch.arange(next_action.size(0)),next_action], weight_left)
          addition_mask = torch.zeros_like(mu)
          addition_mask[torch.arange(mu.shape[0]),next_action] = 1
          weight_left_mask = weight_left >= 0
          result = result + (x.unsqueeze(1) * addition_mask * weight_left_mask.unsqueeze(1))
          weight_left -= x
          
      weight_left_next = weight_left
      result_next = result
      addition_mask_next = torch.zeros_like(mu)
      addition_mask_next[torch.arange(mu.shape[0]),order[:,0]] = 1
      weight_left_next_mask = weight_left_next > 0
      result_next = result_next + (weight_left_next.unsqueeze(1) * weight_left_next_mask.unsqueeze(1) * addition_mask_next)
      discretized_mu = result_next / policy_discretization
      return discretized_mu

@dataclasses.dataclass
class AdamConfig:
  """Adam optimizer related params."""
  b1: float = 0.0
  b2: float = 0.999
  eps: float = 10e-8


@dataclasses.dataclass
class NerdConfig:
  """Nerd related params."""
  beta: float = 2.0
  clip: float = 10_000


class StateRepresentation(str, enum.Enum):
  INFO_SET = "info_set"
  OBSERVATION = "observation"

@dataclasses.dataclass  
class RNaDConfig:
  """Configuration parameters for the RNaDSolver."""
  # The game parameter string including its name and parameters.
  game_name: str
  # The games longer than this value are truncated. Must be strictly positive.
  trajectory_max: int = 10

  # The content of the EnvStep.obs tensor.
  state_representation: StateRepresentation = StateRepresentation.INFO_SET

  # Network configuration.
  policy_network_layers: Sequence[int] = (256, 256)

  # The batch size to use when learning/improving parameters.
  batch_size: int = 256
  # The learning rate for `params`.
  learning_rate: float = 0.00005
  # The config related to the ADAM optimizer used for updating `params`.
  adam: AdamConfig = dataclasses.field(default_factory=AdamConfig)
  # All gradients values are clipped to [-clip_gradient, clip_gradient].
  clip_gradient: float = 10_000
  # The "speed" at which `params_target` is following `params`.
  target_network_avg: float = 0.001

  # RNaD algorithm configuration.
  # Entropy schedule configuration. See EntropySchedule class documentation.
  entropy_schedule_repeats: Sequence[int] = (1,)
  entropy_schedule_size: Sequence[int] = (20_000,)
  # The weight of the reward regularisation term in RNaD.
  eta_reward_transform: float = 0.2
  nerd: NerdConfig = dataclasses.field(default_factory=NerdConfig)
  c_vtrace: float = 1.0

  # Options related to fine tuning of the agent.
  finetune: FineTuning = dataclasses.field(default_factory=FineTuning)

  # The seed that fully controls the randomness.
  seed: int = 42
  
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
    adv_pi = q_vr - torch.sum(pi.squeeze() * q_vr, dim=-1, keepdim=True)
    adv_pi = is_c * adv_pi  # importance sampling correction
    adv_pi = torch.clamp(adv_pi, min=-clip, max=clip)
    adv_pi = adv_pi.detach()

    valid_logit_sum = torch.sum(logit_pi.squeeze() * legal_actions, dim=-1, keepdim=True)
    mean_logit = valid_logit_sum / num_valid_actions

    # Subtract only the mean of the valid logits
    logits = logit_pi.squeeze() - mean_logit

    threshold_center = torch.zeros_like(logits)

    nerd_loss = torch.sum(
        legal_actions *
        apply_force_with_threshold(logits, adv_pi, threshold, threshold_center),
        dim=-1)
    nerd_loss = -renormalize(nerd_loss, valid * (player_ids == k))
    loss_pi_list.append(nerd_loss)
  return torch.sum(torch.stack(loss_pi_list))

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

def get_loss_v(v_list: Sequence[torch.Tensor],
               v_target_list: Sequence[torch.Tensor],
               mask_list: Sequence[torch.Tensor]) -> torch.Tensor:
  """Define the loss function for the critic."""
  assert all(v.shape == v_target.shape for v, v_target in zip(v_list, v_target_list))
  # v_list and v_target_list come with a degenerate trailing dimension,
  # which mask_list tensors do not have.
  assert mask_list[0].shape == v_list[0].shape 
  loss_v_list = []
  for (v_n, v_target, mask) in zip(v_list, v_target_list, mask_list):
    assert v_n.shape[0] == v_target.shape[0]

    loss_v = mask * (v_n - v_target.detach())**2  # Add dim for broadcasting
    normalization = torch.sum(mask)
    loss_v = torch.sum(loss_v) / (normalization + (normalization == 0.0))

    loss_v_list.append(loss_v)
  return torch.sum(torch.stack(loss_v_list))
  
def _legal_policy(logits: torch.Tensor, legal_actions: torch.Tensor) -> torch.Tensor:
  """A soft-max policy that respects legal_actions."""
  assert logits.shape == legal_actions.shape
  # legal_actions is a boolean tensor
  legal_actions = legal_actions.bool()
  # Fiddle a bit to make sure we don't generate NaNs or Inf in the middle.
  l_min = logits.min(axis=-1, keepdims=True)
  logits = torch.where(legal_actions, logits, l_min.values)
  logits -= torch.max(logits, axis=-1, keepdims=True).values
  logits *= legal_actions
  exp_logits = torch.where(legal_actions, torch.exp(logits),
                         0)  # Illegal actions become 0.
  exp_logits_sum = torch.sum(exp_logits, axis=-1, keepdims=True)
  return exp_logits / exp_logits_sum

def legal_log_policy(logits: torch.Tensor,
                     legal_actions: torch.Tensor) -> torch.Tensor:
  """Return the log of the policy on legal action, 0 on illegal action."""
  assert logits.shape == legal_actions.shape
  # logits_masked has illegal actions set to -inf.
  logits_masked = logits + torch.log(legal_actions)
  max_legal_logit = logits_masked.max(axis=-1, keepdims=True).values
  logits_masked = logits_masked - max_legal_logit
  # exp_logits_masked is 0 for illegal actions.
  exp_logits_masked = torch.exp(logits_masked)

  baseline = torch.log(torch.sum(exp_logits_masked, axis=-1, keepdims=True))
  # Subtract baseline from logits. We do not simply return
  #     logits_masked - baseline
  # because that has -inf for illegal actions, or
  #     legal_actions * (logits_masked - baseline)
  # because that leads to 0 * -inf == nan for illegal actions.
  log_policy = torch.multiply(legal_actions,
                            (logits - max_legal_logit - baseline))
  return log_policy


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
class RNaDSolver(policy.Policy):
  def __init__(self, config: RNaDConfig):
    self.config = config
    # Learner and actor step counters.
    self.learner_steps = 0
    self.actor_steps = 0

    self.init()

  def init(self):
    """Initialize the network and losses."""
    # Set seeds for reproducibility
    random_seed_torch = torch.manual_seed(self.config.seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(self.config.seed)
    # Optional: For full reproducibility on CUDA, but can impact performance
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    random_seed_np = np.random.seed(self.config.seed) # Global NumPy seed
    # Create a game and an example of a state.
    self._game = pyspiel.load_game(self.config.game_name)

    self._ex_state = self._play_chance(self._game.new_initial_state())
    obs,_,_,_,_ = self._state_as_env_step(self._ex_state)

   # replay buffer
    self.rb = None

    self.params = ActorCriticNetwork(obs.size().numel(), self.config.policy_network_layers, self._game.num_distinct_actions())

    # The machinery related to updating parameters/learner.
    self._entropy_schedule = EntropySchedule(
        sizes=self.config.entropy_schedule_size,
        repeats=self.config.entropy_schedule_repeats)
    
    # #####
    # create the networks for the prev_policy, target_policy. at the beginning, they are the same
    self.params_target = copy.deepcopy(self.params)
    self.params_prev = copy.deepcopy(self.params)
    self.params_prev_ = copy.deepcopy(self.params)
    self.optimizer = Adam(self.params.parameters(), lr=self.config.learning_rate, betas=(self.config.adam.b1, self.config.adam.b2), eps=self.config.adam.eps)
    
  # def _network_apply_and_post_process(
  #     self, params: Params, env_step: EnvStep) -> chex.Array:
  #   pi, _, _, _ = self.network.apply(params, env_step)
  #   pi = self.config.finetune.post_process_policy(pi, env_step.legal)
  #   return pi
  
  def actor_step(self, env_step_td: TensorDict):
    with torch.no_grad():
      pi, _, _, _ = self.params(env_step_td["obs"], env_step_td["legal"])
      pi = pi / torch.sum(pi, dim=-1, keepdim=True)
      # remove the timestep
      pi = pi.squeeze(1)
    action = np.apply_along_axis(
        lambda x: np.random.choice(range(pi.shape[1]), p=x), axis=-1, arr=pi)
    # TODO(author16): reapply the legal actions mask to bullet-proof sampling.
    action_oh = torch.zeros(pi.shape)
    action_oh[range(pi.shape[0]), action] = 1.0

    # self.rb["action_oh"][:, timestep, :] = action_oh
    # self.rb["policy"][:, timestep, :] = pi
    return action, action_oh, pi
  def step(self):
    """One step of the algorithm, that plays the game and improves params."""
    timesteps = self.collect_batch_trajectory()
    alpha, update_target_net = self._entropy_schedule(self.learner_steps)
    logs = self.update_parameters(timesteps, alpha, self.learner_steps, update_target_net)
    self.learner_steps += 1
    logs.update({
        "actor_steps": self.actor_steps,
        "learner_steps": self.learner_steps,
    })
    # PRINTING
    print("--------------------------------")
    print("Learner steps: ", self.learner_steps)
    print("Actor steps: ", self.actor_steps)
    print("Loss: ", logs["loss"].item())
    print("--------------------------------")
    return logs
  
  
  def action_probabilities(self, state):
    """Computes action probabilities for the current player in state.

    Args:
      state: (pyspiel.State) The state to compute probabilities for.

    Returns:
      (dict) action probabilities for a single batch.
    """
    cur_player = state.current_player()
    legal_actions = state.legal_actions_mask(cur_player)
    env_step = TensorDict()
    obs = torch.tensor(state.information_state_tensor(), dtype=torch.float32)
    env_step["obs"] = obs.reshape(1,*obs.size())
    legal = torch.tensor(legal_actions, dtype=bool)
    env_step["legal"] = legal.reshape(1,*legal.size())
    with torch.no_grad():
      pi, _, _, _ = self.params_target(env_step["obs"], env_step["legal"])
      pi = pi.numpy()
    return {action: pi[action] for action in legal_actions}

  def loss(self, ts: TensorDict, alpha: float,
           learner_steps: int) -> float:
    # pass every timestep to the network using torch vmap
    ts["env_step_td"].batch_size = (self.config.batch_size,self.config.trajectory_max)
    pi, v, log_pi, logit = torch.vmap(self.params, in_dims=(1,1), out_dims=1)(ts["env_step_td"]["obs"], ts["env_step_td"]["legal"])
    policy_pprocessed = self.config.finetune(pi, ts["env_step_td"]["legal"], learner_steps)

    _, v_target, _, _ = torch.vmap(self.params_target, in_dims=1, out_dims=1)(ts["env_step_td"]["obs"], ts["env_step_td"]["legal"])
    _, _, log_pi_prev, _ = torch.vmap(self.params_prev, in_dims=1, out_dims=1)(ts["env_step_td"]["obs"], ts["env_step_td"]["legal"])
    _, _, log_pi_prev_, _ = torch.vmap(self.params_prev_, in_dims=1, out_dims=1)(ts["env_step_td"]["obs"], ts["env_step_td"]["legal"])
    # This line creates the reward transform log(pi(a|x)/pi_reg(a|x)).
    # For the stability reasons, reward changes smoothly between iterations.
    # The mixing between old and new reward transform is a convex combination
    # parametrised by alpha.
    log_policy_reg = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_)

    v_target_list, has_played_list, v_trace_policy_target_list = [], [], []
    for player in range(self._game.num_players()):
      reward = ts["env_step_td"]["rewards"].squeeze()[:,:,player]  # [T, B, Player]
      v_target_, has_played, policy_target_ = v_trace(
          v_target,
          ts["env_step_td"]["valid"].squeeze(),
          ts["env_step_td"]["player_id"].squeeze(),
          ts["policy"].squeeze(),
          policy_pprocessed.squeeze(),
          log_policy_reg.squeeze(),
          player_others(ts["env_step_td"]["player_id"].squeeze(), ts["env_step_td"]["valid"].squeeze(), player),
          ts["action_oh"].squeeze(),
          reward,
          player,
          lambda_=1.0,
          c=self.config.c_vtrace,
          rho=np.inf,
          eta=self.config.eta_reward_transform)
      v_target_list.append(v_target_)
      has_played_list.append(has_played)
      v_trace_policy_target_list.append(policy_target_)
    loss_v = get_loss_v([v.squeeze()] * self._game.num_players(), v_target_list,
                        has_played_list)

    is_vector = torch.ones((self.config.batch_size, self.config.trajectory_max, self._game.num_distinct_actions()))
    importance_sampling_correction = [is_vector] * self._game.num_players()
    # Uses v-trace to define q-values for Nerd
    loss_nerd = get_loss_nerd(
        [logit] * self._game.num_players(), [pi] * self._game.num_players(),
        v_trace_policy_target_list,
        ts["env_step_td"]["valid"].squeeze(),
        ts["env_step_td"]["player_id"].squeeze(),
        ts["env_step_td"]["legal"].squeeze(),
        importance_sampling_correction,
        clip=self.config.nerd.clip,
        threshold=self.config.nerd.beta)
    total_loss = loss_v + loss_nerd
    return total_loss
  
  
  def update_parameters(
      self,
      timestep: TensorDict,
      alpha: float,
      learner_steps: int,
      update_target_net: bool):
    """Update parameters using computed gradients and target network updates."""
    
    # Zero gradients from previous iteration
    self.optimizer.zero_grad()
    
    # Compute loss and perform backward pass
    loss_val = self.loss(timestep, alpha, learner_steps)
    
    # Clip gradients if specified
    if self.config.clip_gradient > 0:
        torch.nn.utils.clip_grad_norm_(self.params.parameters(), self.config.clip_gradient)
    
    # Update main parameters using optimizer (equivalent to optimizer(params, grad) in JAX)
    self.optimizer.step()
    
    # Update target network towards main network (exponential moving average)
    with torch.no_grad():
        for target_param, main_param in zip(self.params_target.parameters(), self.params.parameters()):
            target_param.data.mul_(1 - self.config.target_network_avg).add_(
                main_param.data, alpha=self.config.target_network_avg
            )
    
    # Conditionally roll forward the previous parameters
    if update_target_net:
        # Move params_target -> params_prev and params_prev -> params_prev_
        self.params_prev_.load_state_dict(self.params_prev.state_dict())
        self.params_prev.load_state_dict(self.params_target.state_dict())

    logs = {
        "loss": loss_val,
    }
    return logs
  
  def _batch_of_states_as_env_step(self,
                                   states: Sequence[pyspiel.State]) -> TensorDict:
    env_step_td = TensorDict()
    with torch.no_grad():
      for i, state in enumerate(states):
        obs, legal, player_id, valid, rewards = self._state_as_env_step(state)
        obs = obs.reshape(1,*obs.size())
        legal = legal.reshape(1,*legal.size())
        player_id = player_id.reshape(1,1)
        valid = valid.reshape(1,1)
        rewards = rewards.reshape(1,*rewards.size())
        if "obs" not in env_step_td:
          env_step_td["obs"] = obs
          env_step_td["legal"] = legal
          env_step_td["player_id"] = player_id
          env_step_td["valid"] = valid
          env_step_td["rewards"] = rewards
        else:
          env_step_td["obs"] = torch.cat([env_step_td["obs"], obs], dim=0)
          env_step_td["legal"] = torch.cat([env_step_td["legal"], legal], dim=0)
          env_step_td["player_id"] = torch.cat([env_step_td["player_id"], player_id], dim=0)
          env_step_td["valid"] = torch.cat([env_step_td["valid"], valid], dim=0)
          env_step_td["rewards"] = torch.cat([env_step_td["rewards"], rewards], dim=0)
    return env_step_td
        
  def _batch_of_states_apply_action(
      self, states: Sequence[pyspiel.State],
      actions: torch.Tensor) -> Sequence[pyspiel.State]:
    """Apply a batch of `actions` to a parallel list of `states`."""
    for state, action in zip(states, list(actions)):
      if not state.is_terminal():
        self.actor_steps += 1
        state.apply_action(action)
        self._play_chance(state)
    return states
  
  def collect_batch_trajectory(self) -> TensorDict:
    states = [
        self._play_chance(self._game.new_initial_state())
        for _ in range(self.config.batch_size)
    ]
    timesteps = []
    env_step_td = self._batch_of_states_as_env_step(states)
    
    for _ in range(self.config.trajectory_max):
      timesteps_td = TensorDict()
      a, action_oh, pi = self.actor_step(env_step_td)
      states = self._batch_of_states_apply_action(states, a)
      timesteps_td["env_step_td"] = env_step_td
      timesteps_td["action_oh"] = action_oh
      timesteps_td["policy"] = pi
      timesteps.append(timesteps_td)
      env_step_td = self._batch_of_states_as_env_step(states)
    return torch.stack(timesteps,1)
  def _state_as_env_step(self, state: pyspiel.State) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # A terminal state must be communicated to players, however since
    # it's a terminal state things like the state_representation or
    # the set of legal actions are meaningless and only needed
    # for the sake of creating well a defined trajectory tensor.
    # Therefore the code below:
    # - extracts the rewards
    # - if the state is terminal, uses a dummy other state for other fields.
    rewards = state.returns()

    valid = not state.is_terminal()
    if not valid:
      state = self._ex_state

    if self.config.state_representation == StateRepresentation.OBSERVATION:
      obs = state.observation_tensor()
    elif self.config.state_representation == StateRepresentation.INFO_SET:
      obs = state.information_state_tensor()
    else:
      raise ValueError(
          f"Invalid StateRepresentation: {self.config.state_representation}.")

    # TODO(author16): clarify the story around rewards and valid.
    return torch.tensor(obs, dtype=torch.float32), torch.tensor(state.legal_actions_mask(), dtype=bool), torch.tensor(state.current_player(), dtype=torch.int8), torch.tensor(valid, dtype=bool), torch.tensor(rewards, dtype=torch.float16)
    
  def _play_chance(self, state: pyspiel.State) -> pyspiel.State:
    """Plays the chance nodes until we end up at another type of node.

    Args:
      state: to be updated until it does not correspond to a chance node.
    Returns:
      The same input state object, but updated. The state is returned
      only for convenience, to allow chaining function calls.
    """
    while state.is_chance_node():
      chance_outcome, chance_proba = zip(*state.chance_outcomes())
      action = np.random.choice(chance_outcome, p=chance_proba)
      state.apply_action(action)
    return state

from open_spiel.python.algorithms import exploitability
solver = RNaDSolver(RNaDConfig(game_name="leduc_poker"))
for _ in range(1000):
  solver.step()
  p = policy.tabular_policy_from_callable(solver._game, solver.action_probabilities)
  conv = exploitability.nash_conv(solver._game, p)
  print("Deep CFR - NashConv:", conv)  