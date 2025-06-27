import torch
from NFSPActor import NFSP
from Evaluator import Evaluator
import ray
import pickle
from MLPs import BR_MLP, AVG_MLP
from open_spiel.python import policy
# exploitability
from open_spiel.python.algorithms import exploitability
class NFSPPolicies(policy.Policy):
  """Joint policy constructed from the NFSP agents for evaluation."""

  def __init__(self, env, nfsp_actor):
    game = env
    player_ids = [0, 1]
    super().__init__(game, player_ids)
    self._actor = nfsp_actor
    
  def action_probabilities(self, state):
    cur_player = state.current_player()

    legal_actions = state.legal_actions()
    # Ask the NFSP actor for an action (deterministic in evaluation mode).
    _, probs = ray.get(self._actor.step.remote(state, cur_player, is_evaluation=True))

    # Build a full probability distribution over all legal actions.
    return {a: float(probs[a]) for a in legal_actions}
  
@ray.remote(num_cpus=4, namespace="nfsp")
class Learner:
    def __init__(self, **kwargs):
      self.game = kwargs["game"]
      self.num_players = kwargs["num_players"]
      self.num_actions = kwargs["num_actions"]
      self.learning_rate = kwargs["learning_rate"]
      self.state_representation_size = kwargs["state_representation_size"]
      self.hidden_layers_sizes = kwargs["hidden_layers_sizes"]
      self.batch_size = kwargs["batch_size"]
      self.replay_buffer_capacity = kwargs["replay_buffer_capacity"]
      self.q_networks = [BR_MLP(self.state_representation_size, self.hidden_layers_sizes, self.num_actions) for _ in range(self.num_players)]
      self.q_net_optimizers = [torch.optim.SGD(q_network.parameters(), lr=self.learning_rate) for q_network in self.q_networks]
      self.avg_networks = [AVG_MLP(self.state_representation_size, self.hidden_layers_sizes, self.num_actions) for _ in range(self.num_players)]
      self.avg_net_optimizers = [torch.optim.SGD(avg_network.parameters(), lr=self.learning_rate) for avg_network in self.avg_networks]
      self.num_actors = kwargs["num_actors"]
      self.num_train_episodes = kwargs["num_train_episodes"]
      self.eval_every = kwargs["eval_every"]
      self.epsilon_start = kwargs["epsilon_start"]
      self.epsilon_end = kwargs["epsilon_end"]
      self.epsilon_decay_duration = kwargs["epsilon_decay_duration"]
      self.min_buffer_size_to_learn = kwargs["min_buffer_size_to_learn"]
      self.learn_every = kwargs["learn_every"]
      self.anticipatory_param = kwargs["anticipatory_param"]
      self.reservoir_buffer_capacity = kwargs["reservoir_buffer_capacity"]
      self.update_target_network_every = kwargs["update_target_network_every"]
      # Create a dedicated evaluator actor
      self.evaluator = Evaluator.options(name="evaluator", namespace="nfsp", lifetime="detached").remote(
            self.game, self.num_players, self.num_actions)
      self.actors = []
    def push_networks(self):
      q_net_state_dicts = [q_network.state_dict() for q_network in self.q_networks]
      avg_net_state_dicts = [avg_network.state_dict() for avg_network in self.avg_networks]
      q_opt_state_dicts = [q_net_optimizer.state_dict() for q_net_optimizer in self.q_net_optimizers]
      avg_opt_state_dicts = [avg_net_optimizer.state_dict() for avg_net_optimizer in self.avg_net_optimizers]
      
      return ray.put(q_net_state_dicts), ray.put(avg_net_state_dicts), ray.put(q_opt_state_dicts), ray.put(avg_opt_state_dicts)
    
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
      self.q_networks = [BR_MLP(self.state_representation_size, self.hidden_layers_sizes, self.num_actions) for _ in range(self.num_players)]
      self.avg_networks = [AVG_MLP(self.state_representation_size, self.hidden_layers_sizes, self.num_actions) for _ in range(self.num_players)]
      self.q_net_optimizers = [torch.optim.SGD(q_network.parameters(), lr=self.learning_rate) for q_network in self.q_networks]
      self.avg_net_optimizers = [torch.optim.SGD(avg_network.parameters(), lr=self.learning_rate) for avg_network in self.avg_networks]
      for i in range(self.num_players):
        self.q_networks[i].load_state_dict(checkpoint["q_networks"][i])
        self.avg_networks[i].load_state_dict(checkpoint["avg_networks"][i])
        self.q_net_optimizers[i].load_state_dict(checkpoint["q_net_optimizers"][i])
        self.avg_net_optimizers[i].load_state_dict(checkpoint["avg_net_optimizers"][i])
    
    def aggregate_gradients(self, gradients_list):
      """Aggregate gradients from multiple actors."""
      if not gradients_list or all(not grads for grads in gradients_list):
        return {}
        
      # Get all parameter names from the first non-empty gradient dict
      param_names = None
      for grads in gradients_list:
        if grads:
          param_names = grads.keys()
          break
          
      if param_names is None:
        return {}
        
      aggregated = {}
      num_actors = len([g for g in gradients_list if g])  # Count non-empty gradient dicts
      
      if num_actors == 0:
        return {}
        
      for param_name in param_names:
        # Collect gradients for this parameter from all actors
        param_grads = []
        for grads in gradients_list:
          if param_name in grads and grads[param_name] is not None:
            param_grads.append(grads[param_name])
        
        if param_grads:
          # Average the gradients
          aggregated_grad = torch.stack(param_grads).mean(dim=0)
          aggregated[param_name] = aggregated_grad
          
      return aggregated
    
    def apply_gradients_to_avg_network(self, aggregated_gradients, player_id):
      """Apply aggregated gradients to average policy network using optimizer."""
      if not aggregated_gradients:
        return
        
      # Set the gradients on the network parameters
      for name, param in self.avg_networks[player_id].named_parameters():
        if name in aggregated_gradients:
          param.grad = aggregated_gradients[name].to(param.device)
      
      # Use the optimizer to apply the gradients (respects momentum, weight decay, etc.)
      self.avg_net_optimizers[player_id].step()
      self.avg_net_optimizers[player_id].zero_grad()

    def apply_gradients_to_q_network(self, aggregated_gradients, player_id):
      """Apply aggregated gradients to Q-network using optimizer."""
      if not aggregated_gradients:
        return
        
      # Set the gradients on the network parameters  
      for name, param in self.q_networks[player_id].named_parameters():
        if name in aggregated_gradients:
          param.grad = aggregated_gradients[name].to(param.device)
      
      # Use the optimizer to apply the gradients
      self.q_net_optimizers[player_id].step()
      self.q_net_optimizers[player_id].zero_grad()
          
    def send_networks_to_workers(self):
      """Broadcast the latest parameters to all actors.

      Serialises (ray.put) the parameter state ONLY ONCE and passes the
      resulting ObjectRefs to every actor.  This avoids redundant
      serialisation work and extra copies in the object store – a significant
      overhead when the networks grow larger.
      """

      # Serialise once ➜ single set of ObjectRefs.
      q_nets_ref, avg_nets_ref, q_opt_ref, avg_opt_ref = self.push_networks()

      # Broadcast those references to each actor.
      for actor in self.actors:
        actor.update_networks.remote(
          q_nets_ref,
          avg_nets_ref,
          q_opt_ref,
          avg_opt_ref,
        )
        
    def create_actors(self):
      for actor in range(self.num_actors):
        self.actors.append(
          NFSP.options(
            name=f"nfsp_{actor}",
            lifetime="detached",
            namespace="nfsp"
          ).remote(
                self.game,
                self.num_players,
                self.num_actions,
                self.replay_buffer_capacity,
                self.reservoir_buffer_capacity,
                self.anticipatory_param,
                self.avg_net_optimizers,
                self.avg_networks,
                self.q_net_optimizers,
                self.q_networks,
                self.update_target_network_every,
                self.epsilon_start,
                self.epsilon_end,
                self.epsilon_decay_duration,
                self.batch_size,
                self.min_buffer_size_to_learn,
                self.learn_every
                )
        )
          
    def start(self):
        """Main training loop with the profiling logic removed."""
        send_networks = True
        # Create the remote NFSP actors.
        self.create_actors()
        evaluation_policy = NFSPPolicies(self.game, self.actors[0])
        for iteration in range(self.num_train_episodes*self.num_actors):
            # ------------------------------------------------------------------
            # 1) Distribute the latest parameters to all actors
            # ------------------------------------------------------------------
            if send_networks:
              self.send_networks_to_workers()
              send_networks = False

            # ------------------------------------------------------------------
            # 2) Periodic evaluation + loss querying
            # ------------------------------------------------------------------
            if (iteration + 1) % self.eval_every == 0:
                # 2-a) Evaluation.
                # eval_ref = self.evaluator.comprehensive_evaluation.remote(
                #     self.q_networks,
                #     self.avg_networks,
                #     num_head_to_head_episodes=100,
                # )
                # eval_results = ray.get(eval_ref)
                # print(f"Iteration {iteration+1} - Evaluation Results:\n{eval_results}")
                nash_conv = exploitability.exploitability(self.game, evaluation_policy)
                print(f"Iteration {iteration+1} - Nash Conv: {nash_conv}")
                # 2-b) Fetch per-actor loss values.
                loss_refs = [
                    actor.get_loss.remote(player)
                    for player in range(self.num_players)
                    for actor in self.actors
                ]
                losses = ray.get(loss_refs)
                for player in range(self.num_players):
                    print(f"Training losses (player {player}): {losses[player]}")
                print("----------------------------------------------------")

            # ------------------------------------------------------------------
            # 3) Generate trajectories (self-play)
            # ------------------------------------------------------------------
            grads_output = ray.get([actor.traverse_game.remote(iteration) for actor in self.actors])
            # grads_output[i] = (sl_gradients_list, br_gradients_list) for actor i
            # where sl_gradients_list[player] and br_gradients_list[player] are the gradients for that player
            sl_gradients = [[] for _ in range(self.num_players)]
            br_gradients = [[] for _ in range(self.num_players)]
            for i in range(self.num_actors):
                actor_sl_grads, actor_br_grads = grads_output[i]  # unpack (sl_list, br_list)
                for player in range(self.num_players):
                    if actor_sl_grads[player] is not None:
                        sl_gradients[player].append(actor_sl_grads[player])
                    if actor_br_grads[player] is not None:
                        br_gradients[player].append(actor_br_grads[player])
            
            # ------------------------------------------------------------------
            # 4) Aggregate gradients & apply updates to central networks
            # ------------------------------------------------------------------
            
            for player in range(self.num_players):
                agg_br = self.aggregate_gradients(br_gradients[player])
                agg_sl = self.aggregate_gradients(sl_gradients[player])
                if agg_br and agg_sl:
                    self.apply_gradients_to_avg_network(agg_sl, player)  # SL gradients → AVG network
                    self.apply_gradients_to_q_network(agg_br, player)    # BR gradients → Q network
                    send_networks = True
            if iteration % 1000 == 0:
              print(f"Completed iteration {iteration + 1}/{self.num_train_episodes*self.num_actors}")