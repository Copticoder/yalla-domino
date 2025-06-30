# Copyright 2024
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

from absl import app
from absl import flags
import pyspiel
from Learner import Learner
import ray
FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", int(7e6),
                     "Number of training iterations.")
flags.DEFINE_integer("eval_every", 10000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_boolean("resume_from_checkpoint", False,
                     "Whether to resume training from a checkpoint.")
flags.DEFINE_string("checkpoint_path", None,
                    "Path to specific checkpoint file to resume from. If None, loads latest checkpoint.")

flags.DEFINE_list("hidden_layers_sizes", [128,128],
                 "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Probability of using the RL best response as episode policy.")
flags.DEFINE_integer("batch_size", 256,
                     "Batch size for the DQN.")
flags.DEFINE_integer("num_actors", 3,
                     "Number of actors.")
flags.DEFINE_integer("update_target_network_every", 1000,
                     "Number of steps between updating the target network.")
flags.DEFINE_float("discount_factor", 1.0,
                   "Discount factor for the DQN.")
flags.DEFINE_integer("min_buffer_size_to_learn", 1000,
                     "Minimum buffer size to learn.")
flags.DEFINE_float("epsilon_start", 0.1,
                   "Starting epsilon for the epsilon-greedy policy.")
flags.DEFINE_float("epsilon_end", 0.1,
                   "Ending epsilon for the epsilon-greedy policy.")
flags.DEFINE_integer("epsilon_decay_duration", int(1e4),
                     "Number of steps for the epsilon-greedy policy to decay.")
flags.DEFINE_float("learning_rate", 0.01,
                   "Learning rate for the DQN.")
flags.DEFINE_integer("learn_every", 64,
                     "Number of steps between learning updates.")

# WandB configuration flags
flags.DEFINE_string("wandb_project", "nfsp-training",
                   "WandB project name for logging.")
flags.DEFINE_string("wandb_entity", "ahmed-attia-mbzuai",
                   "WandB entity/team name for logging.")
flags.DEFINE_boolean("enable_wandb", True,
                    "Whether to enable WandB logging.")
flags.DEFINE_boolean("calculate_exploitability", True,
                    "Whether to calculate exploitability and nash conv during evaluation. Set to False for large games where these metrics are computationally prohibitive.")

def main(_):
  if ray.is_initialized():
    ray.shutdown()
  ray.init(runtime_env={"env_vars": {"RAY_DEBUG": "1"}})
  game = "leduc_poker"

  env = pyspiel.load_game(game)
  num_players = env.num_players()
  info_state_size = env.information_state_tensor_size()
  num_actions = env.num_distinct_actions()
  hidden_layers_sizes = [int(x) for x in FLAGS.hidden_layers_sizes]
  learner_kwargs ={
    "game": env,
    "state_representation_size": info_state_size,
    "num_players": num_players,
    "num_actions": num_actions,
    "hidden_layers_sizes": hidden_layers_sizes,
    "learn_every": FLAGS.learn_every,
    "batch_size": FLAGS.batch_size,
    "reservoir_buffer_capacity": FLAGS.reservoir_buffer_capacity,
    "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
    "update_target_network_every": FLAGS.update_target_network_every,
    "discount_factor": FLAGS.discount_factor,
    "min_buffer_size_to_learn": FLAGS.min_buffer_size_to_learn,
    "epsilon_start": FLAGS.epsilon_start,
    "epsilon_end": FLAGS.epsilon_end,
    "epsilon_decay_duration": FLAGS.epsilon_decay_duration,
    "learning_rate": FLAGS.learning_rate,
    "anticipatory_param": FLAGS.anticipatory_param,
    "num_actors": FLAGS.num_actors,
    "num_iterations": FLAGS.num_iterations,
    "eval_every": FLAGS.eval_every,
    "wandb_project": FLAGS.wandb_project,
    "wandb_entity": FLAGS.wandb_entity,
    "enable_wandb": FLAGS.enable_wandb,
    "calculate_exploitability": FLAGS.calculate_exploitability
  }
  learner = Learner.remote(**learner_kwargs)
  
  # Start training with checkpoint support
  print(f"Starting training for {FLAGS.num_iterations} iterations...")
  if FLAGS.resume_from_checkpoint:
    print("Resuming from checkpoint...")
    ray.get(learner.start.remote(resume_from_checkpoint=True, checkpoint_path=FLAGS.checkpoint_path))
  else:
    print("Starting fresh training...")
    ray.get(learner.start.remote())
  
  print("Training completed!")
  ray.shutdown()

if __name__ == "__main__":
  app.run(main) 