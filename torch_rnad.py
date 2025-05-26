from typing import Sequence, Tuple
import numpy as np
import torch
import enum
from open_spiel.python import policy as policy_lib
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
  def forward(self, env_step):
      # Convert numpy inputs from EnvStep to torch tensors
      obs_tensor = env_step["obs"]
      # legal_actions are used as masks, boolean is appropriate.
      legal_actions_tensor = env_step["legal"]

      # Pass through the shared MLP
      x = self.shared_mlp(obs_tensor)
      # Actor: output the policy logits
      logits = self.actor_head(x)
      # Critic: output the state value
      value = self.critic_head(x)

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

    update_target_net = np.logical_and(
        learner_step > 0,
        np.sum(learner_step == iteration_start + iteration_size - 1),
    )
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
    policy = policy.view(policy.shape[0],policy.shape[1],1,-1)
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

def _legal_policy(logits: torch.Tensor, legal_actions: torch.Tensor) -> torch.Tensor:
  """A soft-max policy that respects legal_actions."""
  assert logits.shape == legal_actions.shape
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


  # entropy bonus
  eta_reg_entropy = (-eta *
                     torch.sum(merged_policy * merged_log_policy, axis=-1) *
                     torch.squeeze(player_others, axis=-1))
  eta_log_policy = -eta * merged_log_policy * player_others

  @dataclasses.dataclass(frozen=True)
  class LoopVTraceCarry:
    """The carry of the v-trace scan loop."""
    reward: torch.Tensor
    # The cumulated reward until the end of the episode. Uncorrected (v-trace).
    # Gamma discounted and includes eta_reg_entropy.
    reward_uncorrected: torch.Tensor
    next_value: torch.Tensor
    next_v_target: torch.Tensor
    importance_sampling: torch.Tensor

  init_state_v_trace = LoopVTraceCarry(
      reward=torch.zeros_like(reward[-1]),
      reward_uncorrected=torch.zeros_like(reward[-1]),
      next_value=torch.zeros_like(v[-1]),
      next_v_target=torch.zeros_like(v[-1]),
      importance_sampling=torch.ones_like(policy_ratio[-1]))

  def _loop_v_trace(carry: LoopVTraceCarry, x) -> Tuple[LoopVTraceCarry, Any]:
    (cs, player_id, v, reward, eta_reg_entropy, valid, inv_mu, actions_oh,
     eta_log_policy) = x

    reward_uncorrected = (
        reward + gamma * carry.reward_uncorrected + eta_reg_entropy)
    discounted_reward = reward + gamma * carry.reward

    # V-target:
    our_v_target = (
        v + jnp.expand_dims(
            jnp.minimum(rho, cs * carry.importance_sampling), axis=-1) *
        (jnp.expand_dims(reward_uncorrected, axis=-1) +
         gamma * carry.next_value - v) + lambda_ * jnp.expand_dims(
             jnp.minimum(c, cs * carry.importance_sampling), axis=-1) * gamma *
        (carry.next_v_target - carry.next_value))

    opp_v_target = jnp.zeros_like(our_v_target)
    reset_v_target = jnp.zeros_like(our_v_target)

    # Learning output:
    our_learning_output = (
        v +  # value
        eta_log_policy +  # regularisation
        actions_oh * jnp.expand_dims(inv_mu, axis=-1) *
        (jnp.expand_dims(discounted_reward, axis=-1) + gamma * jnp.expand_dims(
            carry.importance_sampling, axis=-1) * carry.next_v_target - v))

    opp_learning_output = jnp.zeros_like(our_learning_output)
    reset_learning_output = jnp.zeros_like(our_learning_output)

    # State carry:
    our_carry = LoopVTraceCarry(
        reward=jnp.zeros_like(carry.reward),
        next_value=v,
        next_v_target=our_v_target,
        reward_uncorrected=jnp.zeros_like(carry.reward_uncorrected),
        importance_sampling=jnp.ones_like(carry.importance_sampling))
    opp_carry = LoopVTraceCarry(
        reward=eta_reg_entropy + cs * discounted_reward,
        reward_uncorrected=reward_uncorrected,
        next_value=gamma * carry.next_value,
        next_v_target=gamma * carry.next_v_target,
        importance_sampling=cs * carry.importance_sampling)
    reset_carry = init_state_v_trace

    # Invalid turn: init_state_v_trace and (zero target, learning_output)
    # pyformat: disable
    return _where(valid,  # pytype: disable=bad-return-type  # numpy-scalars
                  _where((player_id == player),
                         (our_carry, (our_v_target, our_learning_output)),
                         (opp_carry, (opp_v_target, opp_learning_output))),
                  (reset_carry, (reset_v_target, reset_learning_output)))
    # pyformat: enable

  _, (v_target, learning_output) = lax.scan(
      f=_loop_v_trace,
      init=init_state_v_trace,
      xs=(policy_ratio, player_id, v, reward, eta_reg_entropy, valid, inv_mu,
          actions_oh, eta_log_policy),
      reverse=True)

  return v_target, has_played, learning_output
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
class RNaDSolver(policy_lib.Policy):
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
    # self._loss_and_grad = jax.value_and_grad(self.loss, has_aux=False)
    # #####
    # create the networks for the prev_policy, target_policy. at the beginning, they are the same
    self.params_target = copy.deepcopy(self.params)
    self.params_prev = copy.deepcopy(self.params)
    self.params_prev_ = copy.deepcopy(self.params)
    self.optimizer = Adam(self.params.parameters(), lr=self.config.learning_rate, betas=(self.config.adam.b1, self.config.adam.b2), eps=self.config.adam.eps)
    self.optimizer_target = SGD(self.params_target.parameters(), lr=self.config.target_network_avg)
    
  # def _network_apply_and_post_process(
  #     self, params: Params, env_step: EnvStep) -> chex.Array:
  #   pi, _, _, _ = self.network.apply(params, env_step)
  #   pi = self.config.finetune.post_process_policy(pi, env_step.legal)
  #   return pi
  
  def actor_step(self, env_step_td: TensorDict):
    with torch.no_grad():
      pi, _, _, _ = self.params(env_step_td)
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
    (self.optimizer, self.optimizer_target), logs = self.update_parameters(
        self.optimizer, self.optimizer_target, timesteps, alpha,
        self.learner_steps, update_target_net)
    self.learner_steps += 1
    logs.update({
        "actor_steps": self.actor_steps,
        "learner_steps": self.learner_steps,
    })
    return logs

  def loss(self, ts: TensorDict, alpha: float,
           learner_steps: int) -> float:
    # pass every timestep to the network using torch vmap
    ts["env_step_td"].batch_size = (self.config.batch_size,self.config.trajectory_max)
    pi, v, log_pi, logit = torch.vmap(self.params, in_dims=1, out_dims=1)(ts["env_step_td"])
    policy_pprocessed = self.config.finetune(pi, ts["env_step_td"]["legal"], learner_steps)

    _, v_target, _, _ = torch.vmap(self.params_target, in_dims=1, out_dims=1)(ts["env_step_td"])
    _, _, log_pi_prev, _ = torch.vmap(self.params_prev, in_dims=1, out_dims=1)(ts["env_step_td"])
    _, _, log_pi_prev_, _ = torch.vmap(self.params_prev_, in_dims=1, out_dims=1)(ts["env_step_td"])
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
    loss_v = get_loss_v([v] * self._game.num_players(), v_target_list,
                        has_played_list)

    is_vector = jnp.expand_dims(jnp.ones_like(ts.env.valid), axis=-1)
    importance_sampling_correction = [is_vector] * self._game.num_players()
    # Uses v-trace to define q-values for Nerd
    loss_nerd = get_loss_nerd(
        [logit] * self._game.num_players(), [pi] * self._game.num_players(),
        v_trace_policy_target_list,
        ts.env.valid,
        ts.env.player_id,
        ts.env.legal,
        importance_sampling_correction,
        clip=self.config.nerd.clip,
        threshold=self.config.nerd.beta)
    return loss_v + loss_nerd  # pytype: disable=bad-return-type  # numpy-scalars
  
  def update_parameters(
      self,
      optimizer: torch.optim.Adam,
      optimizer_target: torch.optim.SGD,
      timestep: TensorDict,
      alpha: float,
      learner_steps: int,
      update_target_net: bool):

    loss_val, grad = self.loss( timestep, alpha,
                                         learner_steps)
    # # Update `params`` using the computed gradient.
    # params = optimizer(params, grad)
    # # Update `params_target` towards `params`.
    # params_target = optimizer_target(
    #     params_target, tree.tree_map(lambda a, b: a - b, params_target, params))

    # # Rolls forward the prev and prev_ params if update_target_net is 1.
    # # pyformat: disable
    # params_prev, params_prev_ = jax.lax.cond(
    #     update_target_net,
    #     lambda: (params_target, params_prev),
    #     lambda: (params_prev, params_prev_))
    # # pyformat: enable

    # logs = {
    #     "loss": loss_val,
    # }
    # return (params, params_target, params_prev, params_prev_, optimizer,
            # optimizer_target), logs
  
  def _batch_of_states_as_env_step(self,
                                   states: Sequence[pyspiel.State]) -> TensorDict:
    env_step_td = TensorDict()
    with torch.no_grad():
      for i, state in enumerate(states):
        obs, legal, player_id, valid, rewards = self._state_as_env_step(state)
        obs = obs.reshape(1,1,*obs.size())
        legal = legal.reshape(1,1,*legal.size())
        player_id = player_id.reshape(1,1,1)
        valid = valid.reshape(1,1,1)
        rewards = rewards.reshape(1,1,*rewards.size())
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


solver = RNaDSolver(RNaDConfig(game_name="leduc_poker"))
solver.step()