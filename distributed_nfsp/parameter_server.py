import math 
import torch
from torch import nn
import torch.nn.functional as F
from scipy import stats
from nfsp import NFSP
import ray
import pickle
class SonnetLinear(nn.Module):
  """A Sonnet linear module.

  Always includes biases and only supports ReLU activations.
  """

  def __init__(self, in_size, out_size, activate_relu=True):
    """Creates a Sonnet linear layer.

    Args:
      in_size: (int) number of inputs
      out_size: (int) number of outputs
      activate_relu: (bool) whether to include a ReLU activation layer
    """
    super(SonnetLinear, self).__init__()
    self._activate_relu = activate_relu
    stddev = 1.0 / math.sqrt(in_size)
    mean = 0
    lower = (-2 * stddev - mean) / stddev
    upper = (2 * stddev - mean) / stddev
    # Weight initialization inspired by Sonnet's Linear layer,
    # which cites https://arxiv.org/abs/1502.03167v3
    # pytorch default: initialized from
    # uniform(-sqrt(1/in_features), sqrt(1/in_features))
    self._weight = nn.Parameter(
        torch.Tensor(
            stats.truncnorm.rvs(
                lower, upper, loc=mean, scale=stddev, size=[out_size,
                                                            in_size])))
    self._bias = nn.Parameter(torch.zeros([out_size]))

  def forward(self, tensor):
    y = F.linear(tensor, self._weight, self._bias)
    return F.relu(y) if self._activate_relu else y


class MLP(nn.Module):
  """A simple network built from nn.linear layers."""

  def __init__(self,
               input_size,
               hidden_sizes,
               output_size,
               activate_final=False):
    """Create the MLP.

    Args:
      input_size: (int) number of inputs
      hidden_sizes: (list) sizes (number of units) of each hidden layer
      output_size: (int) number of outputs
      activate_final: (bool) should final layer should include a ReLU
    """

    super(MLP, self).__init__()
    self._layers = []
    # Hidden layers
    for size in hidden_sizes:
      self._layers.append(SonnetLinear(in_size=input_size, out_size=size))
      input_size = size
    # Output layer
    self._layers.append(
        SonnetLinear(
            in_size=input_size,
            out_size=output_size,
            activate_relu=activate_final))

    self.model = nn.ModuleList(self._layers)

  def forward(self, x):
    for layer in self.model:
      x = layer(x)
    return x
@ray.remote(num_cpus=1)
class ParameterServer:
    def __init__(self, game, state_representation_size, num_actions, hidden_layers_sizes, batch_size, replay_buffer_capacity, update_target_network_every, learn_every, discount_factor, min_buffer_size_to_learn, epsilon_start, epsilon_end, epsilon_decay_duration, learning_rate, anticipatory_param, num_actors):
        self.game = game
        self.num_players = 2
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.state_representation_size = state_representation_size
        self.hidden_layers_sizes = hidden_layers_sizes
        self.q_networks = [MLP(state_representation_size, hidden_layers_sizes, num_actions) for _ in range(self.num_players)]
        self.q_net_optimizers = [torch.optim.SGD(q_network.parameters(), lr=learning_rate) for q_network in self.q_networks]
        self.avg_networks = [MLP(state_representation_size, hidden_layers_sizes, num_actions) for _ in range(self.num_players)]
        self.avg_net_optimizers = [torch.optim.SGD(avg_network.parameters(), lr=learning_rate) for avg_network in self.avg_networks]
        self.batch_size = batch_size
        self.replay_buffer_capacity = replay_buffer_capacity
        self.update_target_network_every = update_target_network_every
        self.learn_every = learn_every
        self.discount_factor = discount_factor
        self.min_buffer_size_to_learn = min_buffer_size_to_learn
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_duration = epsilon_decay_duration
        self.anticipatory_param = anticipatory_param
        self.num_actors = num_actors
        self.actors = []
    def push_networks(self):
      return ray.put(self.q_networks), ray.put(self.avg_networks), ray.put(self.q_net_optimizers), ray.put(self.avg_net_optimizers)
    
    def save_checkpoint(self):
      # checkpoint the networks, epsilon, and optimizers
      checkpoint = {
        "epsilon": self.epsilon,
        "q_networks": [q_network.state_dict() for q_network in self.q_networks],
        "avg_networks": [avg_network.state_dict() for avg_network in self.avg_networks],
        "q_net_optimizers": [q_net_optimizer.state_dict() for q_net_optimizer in self.q_net_optimizers],
        "avg_net_optimizers": [avg_net_optimizer.state_dict() for avg_net_optimizer in self.avg_net_optimizers],
      }
      # save the checkpoint to a file
      with open("checkpoint.pkl", "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
      
    def load_checkpoint(self):
      with open("checkpoint.pkl", "rb") as f:
        checkpoint = pickle.load(f)
      self.epsilon = checkpoint["epsilon"]
      self.q_networks = [MLP(self.state_representation_size, self.hidden_layers_sizes, self.num_actions) for _ in range(self.num_players)]
      self.avg_networks = [MLP(self.state_representation_size, self.hidden_layers_sizes, self.num_actions) for _ in range(self.num_players)]
      self.q_net_optimizers = [torch.optim.SGD(q_network.parameters(), lr=self.learning_rate) for q_network in self.q_networks]
      self.avg_net_optimizers = [torch.optim.SGD(avg_network.parameters(), lr=self.learning_rate) for avg_network in self.avg_networks]
      for i in range(self.num_players):
        self.q_networks[i].load_state_dict(checkpoint["q_networks"][i])
        self.avg_networks[i].load_state_dict(checkpoint["avg_networks"][i])
        self.q_net_optimizers[i].load_state_dict(checkpoint["q_net_optimizers"][i])
        self.avg_net_optimizers[i].load_state_dict(checkpoint["avg_net_optimizers"][i])
    
    def apply_gradients_to_avg_network(self, gradients_br, player_id):
      for name, param in self.avg_networks[player_id].named_parameters():
        if param.grad is not None:
          param.data.add_(gradients_br[name], alpha=-self._learning_rate)

    def apply_gradients_to_q_network(self, gradients_rl, player_id):
      for name, param in self.q_networks[player_id].named_parameters():
        if param.grad is not None:
          param.data.add_(gradients_rl[name], alpha=-self._learning_rate)
          
    def send_networks_to_workers(self):
      for actor in range(self.num_actors//self.num_players):
        q_nets_ref,avg_nets_ref,q_net_opts_ref,avg_net_opts_ref = self.push_networks()
        self.actors[actor].update_networks.remote(q_nets_ref,avg_nets_ref,q_net_opts_ref,avg_net_opts_ref)
        
    def create_actors(self):
      for actor in range(self.num_actors//self.num_players):
        self.actors.append(
          NFSP.options(
            name=f"nfsp_{actor}",
            lifetime="detached"
          ).remote(
                    self.num_players,
                    self.state_representation_size,
                    self.num_actions,
                    self.hidden_layers_sizes,
                    self.replay_buffer_capacity,
                    self.anticipatory_param,
                    batch_size=self.batch_size,
                    q_networks=self.q_networks,
                    avg_networks=self.avg_networks,
                    q_net_optimizers=self.q_net_optimizers,
                    avg_net_optimizers=self.avg_net_optimizers,
                )
        )
          
    def start(self):
        # create n/2 actors for each player
        self.create_actors()
        for ep in range(self.num_train_episodes):
            # distribute networks and optimizers to workers
            self.send_networks_to_workers()
            if (ep + 1) % self.eval_every == 0:
                losses = [agent.remote().loss for agent in self.actors]
                print("Episode %s - Losses %s", ep + 1, losses)
                # expl = exploitability.exploitability(env.game, eval_policy_avg)
                # logging.info("Episode %s - Exploitability (avg policy): %.4f", ep + 1, expl)
                print("----------------------------------------------------")
            # start episodes 
            for actor in range(self.num_actors//self.num_players):
              flags = ray.get(self.actors[actor].traverse_game.remote())
            # learn
            # Launch asynchronous learn_br_rl calls for every (actor, player) pair
            pending_refs = {}
            for actor in range(self.num_actors // self.num_players):
              for player in range(self.num_players):
                ref = self.actors[actor].learn_br_rl.remote(player)
                pending_refs[ref] = player  # keep track of the originating player
            gradients_br = {k: [] for k in range(self.num_players)}
            gradients_rl = {k: [] for k in range(self.num_players)}
            # Retrieve and apply gradients as soon as they are ready
            while pending_refs:
              ready_refs, _ = ray.wait(list(pending_refs.keys()), num_returns=1)
              for ready_ref in ready_refs:
                player = pending_refs.pop(ready_ref)
                # we retrieve two gradients, one for the br and one for the rl
                gradients_br, gradients_rl = ray.get(ready_ref)
                gradients_br[player].append(gradients_br)
                gradients_rl[player].append(gradients_rl)
            
            for player in range(self.num_players):
              self.apply_gradients_to_avg_network(gradients_br[player], player)
              self.apply_gradients_to_q_network(gradients_br[player], player)
            # update target networks
            for actor in range(self.num_actors//self.num_players):
              self.actors[actor].update_target_networks.remote()