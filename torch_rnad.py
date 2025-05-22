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
import gc

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
  def forward(self, env_step, timestep):
      # Convert numpy inputs from EnvStep to torch tensors
      # Assuming network weights are float32, so convert obs to float32.
      obs_tensor = env_step["obs"][:,timestep,:]
      # legal_actions are used as masks, boolean is appropriate.
      legal_actions_tensor = env_step["legal"][:,timestep,:]

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
    do_finetune = torch.logical_and(self.from_learner_steps >= 0,
                                  learner_steps > self.from_learner_steps)

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
    policy = self._discretize(policy)
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
        (torch.max(policy, axis=-1, keepdims=True) < self.policy_threshold))
    return mask * policy / torch.sum(mask * policy, axis=-1, keepdims=True)

  def _discretize(self, policy: torch.Tensor) -> torch.Tensor:
    """Round all action probabilities to a multiple of 1/self.discretize."""
    if self.policy_discretization <= 0:
      return policy

    # The unbatched/single policy case:
    if len(policy.shape) == 1:
      return self._discretize_single(policy)

    # policy may be [B, A] or [T, B, A], etc. Thus add hk.BatchApply.
    dims = len(policy.shape) - 1

    # TODO(author18): avoid mixing vmap and BatchApply since the two could
    # be folded into either a single BatchApply or a sequence of vmaps, but
    # not the mix.
    vmapped = torch.vmap(self._discretize_single)
    policy = hk.BatchApply(vmapped, num_dims=dims)(policy)

    return policy

  def _discretize_single(self, mu: torch.Tensor) -> torch.Tensor:
    """A version of self._discretize but for the unbatched data."""
    # TODO(author18): try to merge _discretize and _discretize_single
    # into one function that handles both batched and unbatched cases.
    if len(mu.shape) == 2:
      mu_ = torch.squeeze(mu, axis=0)
    else:
      mu_ = mu
    n_actions = mu_.shape[-1]
    roundup = torch.ceil(mu_ * self.policy_discretization).astype(torch.int32)
    result = torch.zeros_like(mu_)
    order = torch.argsort(-mu_)  # Indices of descending order.
    weight_left = self.policy_discretization

    def f_disc(i, order, roundup, weight_left, result):
      x = torch.minimum(roundup[order[i]], weight_left)
      result = torch.where(weight_left >= 0, result.at[order[i]].add(x),
                               result)
      weight_left -= x
      return i + 1, order, roundup, weight_left, result

    def f_scan_scan(carry, x):
      i, order, roundup, weight_left, result = carry
      i_next, order_next, roundup_next, weight_left_next, result_next = f_disc(
          i, order, roundup, weight_left, result)
      carry_next = (i_next, order_next, roundup_next, weight_left_next,
                    result_next)
      return carry_next, x

    (_, _, _, weight_left_next, result_next), _ = torch.lax.scan(
        f_scan_scan,
        init=(torch.as_tensor(0), order, roundup, weight_left, result),
        xs=None,
        length=n_actions)

    result_next = torch.where(weight_left_next > 0,
                            result_next.at[order[0]].add(weight_left_next),
                            result_next)
    if len(mu.shape) == 2:
      result_next = torch.expand_dims(result_next, axis=0)
    return result_next / self.policy_discretization

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
  adam: AdamConfig = AdamConfig()
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
  nerd: NerdConfig = NerdConfig()
  c_vtrace: float = 1.0

  # Options related to fine tuning of the agent.
  finetune: FineTuning = FineTuning()

  # The seed that fully controls the randomness.
  seed: int = 42
import pyspiel

@dataclasses.dataclass
class EnvStep:
  """Holds the tensor data representing the current game state."""
  # Indicates whether the state is a valid one or just a padding. Shape: [...]
  # The terminal state being the first one to be marked !valid.
  # All other tensors in EnvStep contain data, but only for valid timesteps.
  # Once !valid the data needs to be ignored, since it's a duplicate of
  # some other previous state.
  # The rewards is the only exception that contains reward values
  # in the terminal state, which is marked !valid.
  # TODO: This is a confusion point and would need to be clarified.
  valid: np.ndarray = dataclasses.field(default_factory=lambda: np.array([], dtype=bool))
  # The single tensor representing the state observation. Shape: [..., ??]
  obs: np.ndarray = dataclasses.field(default_factory=lambda: np.array([], dtype=np.float64))
  # The legal actions mask for the current player. Shape: [..., A]
  legal: np.ndarray = dataclasses.field(default_factory=lambda: np.array([], dtype=np.int8))
  # The current player id as an int. Shape: [...]
  player_id: np.ndarray = dataclasses.field(default_factory=lambda: np.array([], dtype=np.int32))
  # The rewards of all the players. Shape: [..., P]
  rewards: np.ndarray = dataclasses.field(default_factory=lambda: np.array([], dtype=np.float64))


@dataclasses.dataclass
class ActorStep:
  """The actor step tensor summary."""
  # The action (as one-hot) of the current player. Shape: [..., A]
  action_oh: np.ndarray = dataclasses.field(default_factory=lambda: np.array([], dtype=np.int8))
  # The policy of the current player. Shape: [..., A]
  policy: np.ndarray = dataclasses.field(default_factory=lambda: np.array([], dtype=np.float64))
  # The rewards of all the players. Shape: [..., P]
  # Note - these are rewards obtained *after* the actor step, and thus
  # these are the same as EnvStep.rewards visible before the *next* step.
  rewards: np.ndarray = dataclasses.field(default_factory=lambda: np.array([], dtype=np.float64))


@dataclasses.dataclass
class TimeStep:
  """The tensor data for one game transition (env_step, actor_step)."""
  env: EnvStep = EnvStep()
  actor: ActorStep = ActorStep()

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
    self.rb = {
      "obs": torch.zeros(self.config.batch_size, self.config.trajectory_max, obs.size().numel()),
      "valid": torch.zeros(self.config.batch_size, self.config.trajectory_max, 1),
      "player_id": torch.zeros(self.config.batch_size, self.config.trajectory_max, 1),
      "rewards": torch.zeros(self.config.batch_size, self.config.trajectory_max, self._game.num_players()),
      "legal": torch.zeros(self.config.batch_size, self.config.trajectory_max, self._game.num_distinct_actions(), dtype=bool),
      "action_oh": torch.zeros(self.config.batch_size, self.config.trajectory_max, self._game.num_distinct_actions(), dtype=bool),
      "policy": torch.zeros(self.config.batch_size, self.config.trajectory_max, self._game.num_distinct_actions(), dtype=float),
    }

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
  
  def actor_step(self, timestep: int):
    with torch.no_grad():
      pi, _, _, _ = self.params(self.rb, timestep)
      pi = pi / torch.sum(pi, dim=-1, keepdim=True)

    action = np.apply_along_axis(
        lambda x: np.random.choice(range(pi.shape[1]), p=x), axis=-1, arr=pi)
    # TODO(author16): reapply the legal actions mask to bullet-proof sampling.
    action_oh = torch.zeros(pi.shape)
    action_oh[range(pi.shape[0]), action] = 1.0

    self.rb["action_oh"][:, timestep, :] = action_oh
    self.rb["policy"][:, timestep, :] = pi
    return action
  def step(self):
    """One step of the algorithm, that plays the game and improves params."""
    timestep = self.collect_batch_trajectory()
    alpha, update_target_net = self._entropy_schedule(self.learner_steps)
    (self.params, self.params_target, self.params_prev, self.params_prev_,
    self.optimizer, self.optimizer_target), logs = self.update_parameters(
        self.params, self.params_target, self.params_prev, self.params_prev_,
        self.optimizer, self.optimizer_target, timestep, alpha,
        self.learner_steps, update_target_net)
    self.learner_steps += 1
    logs.update({
        "actor_steps": self.actor_steps,
        "learner_steps": self.learner_steps,
    })
    return logs
  
  def loss(self, params: ActorCriticNetwork, params_target: ActorCriticNetwork, params_prev: ActorCriticNetwork,
           params_prev_: ActorCriticNetwork, ts: TimeStep, alpha: float,
           learner_steps: int) -> float:
    print(ts)
    batched_ts_obs = ts.env.obs.repeat(self.config.batch_size, 1)
    pi, v, log_pi, logit = self.params(batched_ts_obs)
    policy_pprocessed = self.config.finetune(pi, ts.env.legal, learner_steps)

    _, v_target, _, _ = self.params_target(batched_ts_obs)
    _, _, log_pi_prev, _ = self.params_prev(batched_ts_obs)
    _, _, log_pi_prev_, _ = self.params_prev_(batched_ts_obs)
    # # This line creates the reward transform log(pi(a|x)/pi_reg(a|x)).
    # # For the stability reasons, reward changes smoothly between iterations.
    # # The mixing between old and new reward transform is a convex combination
    # # parametrised by alpha.
    # log_policy_reg = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_)

    # v_target_list, has_played_list, v_trace_policy_target_list = [], [], []
    # for player in range(self._game.num_players()):
    #   reward = ts.actor.rewards[:, :, player]  # [T, B, Player]
    #   v_target_, has_played, policy_target_ = v_trace(
    #       v_target,
    #       ts.env.valid,
    #       ts.env.player_id,
    #       ts.actor.policy,
    #       policy_pprocessed,
    #       log_policy_reg,
    #       _player_others(ts.env.player_id, ts.env.valid, player),
    #       ts.actor.action_oh,
    #       reward,
    #       player,
    #       lambda_=1.0,
    #       c=self.config.c_vtrace,
    #       rho=np.inf,
    #       eta=self.config.eta_reward_transform)
    #   v_target_list.append(v_target_)
    #   has_played_list.append(has_played)
    #   v_trace_policy_target_list.append(policy_target_)
    # loss_v = get_loss_v([v] * self._game.num_players(), v_target_list,
    #                     has_played_list)

    # is_vector = jnp.expand_dims(jnp.ones_like(ts.env.valid), axis=-1)
    # importance_sampling_correction = [is_vector] * self._game.num_players()
    # # Uses v-trace to define q-values for Nerd
    # loss_nerd = get_loss_nerd(
    #     [logit] * self._game.num_players(), [pi] * self._game.num_players(),
    #     v_trace_policy_target_list,
    #     ts.env.valid,
    #     ts.env.player_id,
    #     ts.env.legal,
    #     importance_sampling_correction,
    #     clip=self.config.nerd.clip,
    #     threshold=self.config.nerd.beta)
    # return loss_v + loss_nerd  # pytype: disable=bad-return-type  # numpy-scalars
  
  def update_parameters(
      self,
      params: ActorCriticNetwork,
      params_target: ActorCriticNetwork,
      params_prev: ActorCriticNetwork,
      params_prev_: ActorCriticNetwork,
      optimizer: torch.optim.Adam,
      optimizer_target: torch.optim.SGD,
      timestep: TimeStep,
      alpha: float,
      learner_steps: int,
      update_target_net: bool):

    loss_val, grad = self.loss(params, params_target, params_prev,
                                         params_prev_, timestep, alpha,
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
                                   states: Sequence[pyspiel.State], timestep: int) -> EnvStep:
    with torch.no_grad():
      for i, state in enumerate(states):
        obs, legal, player_id, valid, rewards = self._state_as_env_step(state)
        self.rb["obs"][i, timestep, :] = obs
        self.rb["legal"][i, timestep, :] = legal
        self.rb["player_id"][i, timestep, :] = player_id
        self.rb["rewards"][i, timestep, :] = rewards
        self.rb["valid"][i, timestep, :] = valid

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
  # [N,T,B]
  def collect_batch_trajectory(self) -> TimeStep:
    states = [
        self._play_chance(self._game.new_initial_state())
        for _ in range(self.config.batch_size)
    ]
    timestep = 0
    self._batch_of_states_as_env_step(states, timestep)
    for _ in range(self.config.trajectory_max):
      a = self.actor_step(timestep)

      states = self._batch_of_states_apply_action(states, a)
      timestep += 1
      self._batch_of_states_as_env_step(states, timestep)
      
  def _state_as_env_step(self, state: pyspiel.State) -> EnvStep:
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
    return torch.tensor(obs), torch.tensor(state.legal_actions_mask()), torch.tensor(state.current_player()), torch.tensor(valid), torch.tensor(rewards)
    
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


solver = RNaDSolver(RNaDConfig(game_name="kuhn_poker"))
solver.step()