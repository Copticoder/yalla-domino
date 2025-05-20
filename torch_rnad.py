from typing import Sequence, Tuple
import numpy as np
import torch
from absl.testing import absltest
from absl.testing import parameterized
from open_spiel.python import policy as policy_lib

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
    torch.manual_seed(self.config.seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(self.config.seed)
    # Optional: For full reproducibility on CUDA, but can impact performance
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(self.config.seed) # Global NumPy seed

    # Create a game and an example of a state.
    self._game = pyspiel.load_game(self.config.game_name)
    self._ex_state = self._play_chance(self._game.new_initial_state())

    # The network.
    def network(
        env_step: EnvStep
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
      mlp_torso = hk.nets.MLP(
          self.config.policy_network_layers, activate_final=True
      )
      torso = mlp_torso(env_step.obs)

      mlp_policy_head = hk.nets.MLP([self._game.num_distinct_actions()])
      logit = mlp_policy_head(torso)

      mlp_policy_value = hk.nets.MLP([1])
      v = mlp_policy_value(torso)

      pi = _legal_policy(logit, env_step.legal)
      log_pi = legal_log_policy(logit, env_step.legal)
      return pi, v, log_pi, logit

    self.network = hk.without_apply_rng(hk.transform(network))

    # The machinery related to updating parameters/learner.
    self._entropy_schedule = EntropySchedule(
        sizes=self.config.entropy_schedule_size,
        repeats=self.config.entropy_schedule_repeats)
    self._loss_and_grad = jax.value_and_grad(self.loss, has_aux=False)

    # Create initial parameters.
    env_step = self._state_as_env_step(self._ex_state)
    key = self._next_rng_key()  # Make sure to use the same key for all.
    self.params = self.network.init(key, env_step)
    self.params_target = self.network.init(key, env_step)
    self.params_prev = self.network.init(key, env_step)
    self.params_prev_ = self.network.init(key, env_step)

    # Parameter optimizers.
    self.optimizer = optax_optimizer(
        self.params,
        optax.chain(
            optax.scale_by_adam(
                eps_root=0.0,
                **self.config.adam,
            ), optax.scale(-self.config.learning_rate),
            optax.clip(self.config.clip_gradient)))
    self.optimizer_target = optax_optimizer(
        self.params_target, optax.sgd(self.config.target_network_avg))

  def init(self):
    pass
