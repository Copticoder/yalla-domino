import torch
from NFSPActor import NFSP
from Evaluator import Evaluator
import ray
import pickle
from MLPs import BR_MLP, AVG_MLP

@ray.remote(num_cpus=2, namespace="nfsp")
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
      breakpoint()
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
          aggregated[param_name] = torch.stack(param_grads).mean(dim=0)
          
      return aggregated
    
    def apply_gradients_to_avg_network(self, aggregated_gradients, player_id):
      """Apply aggregated gradients to average policy network."""
      if not aggregated_gradients:
        return
        
      with torch.no_grad():
        for name, param in self.avg_networks[player_id].named_parameters():
          if name in aggregated_gradients:
            # Apply gradient with learning rate
            param.data.add_(aggregated_gradients[name], alpha=-self.learning_rate)

    def apply_gradients_to_q_network(self, aggregated_gradients, player_id):
      """Apply aggregated gradients to Q-network."""
      if not aggregated_gradients:
        return
        
      with torch.no_grad():
        for name, param in self.q_networks[player_id].named_parameters():
          if name in aggregated_gradients:
            # Apply gradient with learning rate
            param.data.add_(aggregated_gradients[name], alpha=-self.learning_rate)
          
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

        # Create the remote NFSP actors.
        self.create_actors()
        for iteration in range(self.num_train_episodes*self.num_actors):
            # ------------------------------------------------------------------
            # 1) Distribute the latest parameters to all actors
            # ------------------------------------------------------------------
            self.send_networks_to_workers()

            # ------------------------------------------------------------------
            # 2) Periodic evaluation + loss querying
            # ------------------------------------------------------------------
            if (iteration + 1) % self.eval_every == 0:
                # 2-a) Evaluation.
                eval_ref = self.evaluator.comprehensive_evaluation.remote(
                    self.q_networks,
                    self.avg_networks,
                    num_head_to_head_episodes=100,
                )
                eval_results = ray.get(eval_ref)
                print(f"Iteration {iteration+1} - Evaluation Results:\n{eval_results}")

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
            gradients_sl_by_player, gradients_br_by_player = ray.get([actor.traverse_game.remote(iteration) for actor in self.actors])
            breakpoint()
            print(f"Gradients SL: {gradients_sl_by_player}")
            print(f"Gradients BR: {gradients_br_by_player}")
            
            # ------------------------------------------------------------------
            # 4) Aggregate gradients & apply updates to central networks
            # ------------------------------------------------------------------
            
            for player in range(self.num_players):
                agg_br = self.aggregate_gradients(gradients_br_by_player[player])
                agg_sl = self.aggregate_gradients(gradients_sl_by_player[player])
                if agg_br:
                    self.apply_gradients_to_avg_network(agg_br, player)
                if agg_sl:
                    self.apply_gradients_to_q_network(agg_sl, player)

            # ------------------------------------------------------------------
            # 5) Push updated parameters back to the actors
            # ------------------------------------------------------------------
            self.send_networks_to_workers()
            print(f"Completed iteration {iteration + 1}/{self.num_train_episodes*self.num_actors}")