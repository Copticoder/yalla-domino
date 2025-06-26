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

from open_spiel.python import policy
import pyspiel
from open_spiel.python.algorithms import exploitability
from nfsp import NFSP
from parameter_server import ParameterServer
import ray

# class NFSPPolicies(policy.Policy):
#   """Joint policy constructed from the NFSP agents for evaluation."""

#   def __init__(self, env, nfsp_agents, mode):
#     game = env.game
#     player_ids = [0, 1]
#     super().__init__(game, player_ids)
#     self._agents = nfsp_agents
#     self._mode = mode
#     self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

#   def action_probabilities(self, state, player_id=None):
#     cur_player = state.current_player()
#     legal_actions = state.legal_actions(cur_player)

#     self._obs["current_player"] = cur_player
#     self._obs["info_state"][cur_player] = state.information_state_tensor(cur_player)
#     self._obs["legal_actions"][cur_player] = legal_actions

#     info_state = rl_environment.TimeStep(
#         observations=self._obs, rewards=None, discounts=None, step_type=None
#     )

#     with self._agents[cur_player].temp_mode_as(self._mode):
#       p = self._agents[cur_player].step(info_state, is_evaluation=True).probs
#     return {action: p[action] for action in legal_actions}
ray.init(runtime_env={"env_vars": {"RAY_DEBUG_POST_MORTEM": "0"}})

num_train_episodes = int(3e6)
eval_every = 5000
hidden_layers_sizes = [128, 128]
replay_buffer_capacity = int(2e6)
anticipatory_param = 0.1
num_actors = 1
epsilon_start = 0.08
epsilon_end = 0.001
learning_rate = 0.01
batch_size = 128
num_players = 2
discount_factor = 1
min_buffer_size_to_learn = 128
update_target_network_every = 64
game = pyspiel.load_game("kuhn_poker")
info_state_size = game.information_state_tensor_size()
num_actions = game.num_distinct_actions()
parameter_server = ParameterServer.remote(
    game,
    info_state_size,
    num_actions,
    hidden_layers_sizes,
    batch_size,
    replay_buffer_capacity,
    update_target_network_every,
    discount_factor,
    min_buffer_size_to_learn,
    epsilon_start,
    epsilon_end,
    1000,
    learning_rate,
    anticipatory_param,
    num_actors,
    num_train_episodes,
    eval_every,
)

ray.get(parameter_server.start.remote())