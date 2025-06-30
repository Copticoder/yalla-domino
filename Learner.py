import torch
from NFSPActor import NFSP
from Evaluator import Evaluator
import ray
import pickle
from MLPs import BR_MLP, AVG_MLP
from open_spiel.python import policy
# exploitability
from open_spiel.python.algorithms import exploitability
import numpy as np
from NFSPActor import MODE
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

    # Debug: Print action probabilities for the first few evaluations
    if hasattr(self, '_debug_count'):
        self._debug_count += 1
    else:
        self._debug_count = 1
    
    if self._debug_count <= 5:  # Only print for first few calls
        print(f"Debug - State: {state}, Player: {cur_player}, Action probs: {probs}")

    # Build a full probability distribution over all legal actions.
    return {a: float(probs[a]) for a in legal_actions}
  
@ray.remote(num_cpus=6, namespace="nfsp")
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
      self.num_iterations = kwargs["num_iterations"]
      self.eval_every = kwargs["eval_every"]
      self.epsilon_start = kwargs["epsilon_start"]
      self.epsilon_end = kwargs["epsilon_end"]
      self.epsilon_decay_duration = kwargs["epsilon_decay_duration"]
      self.min_buffer_size_to_learn = kwargs["min_buffer_size_to_learn"]
      self.learn_every = kwargs["learn_every"]
      self.anticipatory_param = kwargs["anticipatory_param"]
      self.reservoir_buffer_capacity = kwargs["reservoir_buffer_capacity"]
      self.update_target_network_every = kwargs["update_target_network_every"]
      
      # WandB configuration
      self.wandb_project = kwargs.get("wandb_project", "nfsp-training")
      self.wandb_entity = kwargs.get("wandb_entity", None)
      self.enable_wandb = kwargs.get("enable_wandb", True)
      self.calculate_exploitability = kwargs.get("calculate_exploitability", True)
      
      # Prepare training configuration for wandb
      self.training_config = kwargs
      
      # Create a dedicated evaluator actor
      self.evaluator = Evaluator.options(name="evaluator", namespace="nfsp", lifetime="detached").remote(
            self.game, self.num_players, self.num_actions,
            self.wandb_project, self.wandb_entity, self.enable_wandb, 
            self.calculate_exploitability, self.training_config)
      self.actors = []
      
    # ---------------------------------------------------------------------------
    # Checkpoint and Actor Management Methods
    # ---------------------------------------------------------------------------
    
    def save_checkpoint(self, iteration):
      # checkpoint the networks, optimizers, and current iteration
      checkpoint = {
        "iteration": iteration,
        "q_networks": [q_network.state_dict() for q_network in self.q_networks],
        "avg_networks": [avg_network.state_dict() for avg_network in self.avg_networks],
        "q_net_optimizers": [q_net_optimizer.state_dict() for q_net_optimizer in self.q_net_optimizers],
        "avg_net_optimizers": [avg_net_optimizer.state_dict() for avg_net_optimizer in self.avg_net_optimizers],
      }
      
      # Create checkpoints directory if it doesn't exist
      import os
      os.makedirs("checkpoints", exist_ok=True)
      
      # save the checkpoint to a file
      checkpoint_filename = f"checkpoints/checkpoint_iter_{iteration}.pkl"
      with open(checkpoint_filename, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
      print(f"Checkpoint saved at iteration {iteration}: {checkpoint_filename}")
      
      # Also save as latest checkpoint for easy resuming
      latest_checkpoint_path = "checkpoints/checkpoint_latest.pkl"
      with open(latest_checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_checkpoint(self, checkpoint_path="checkpoints/checkpoint_latest.pkl"):
      print(f"Loading checkpoint from {checkpoint_path}")
      with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
      
      # Restore iteration
      start_iteration = checkpoint["iteration"]
      
      # Recreate networks and optimizers
      self.q_networks = [BR_MLP(self.state_representation_size, self.hidden_layers_sizes, self.num_actions) for _ in range(self.num_players)]
      self.avg_networks = [AVG_MLP(self.state_representation_size, self.hidden_layers_sizes, self.num_actions) for _ in range(self.num_players)]
      self.q_net_optimizers = [torch.optim.SGD(q_network.parameters(), lr=self.learning_rate) for q_network in self.q_networks]
      self.avg_net_optimizers = [torch.optim.SGD(avg_network.parameters(), lr=self.learning_rate) for avg_network in self.avg_networks]
      
      # Load saved states
      for i in range(self.num_players):
        self.q_networks[i].load_state_dict(checkpoint["q_networks"][i])
        self.avg_networks[i].load_state_dict(checkpoint["avg_networks"][i])
        self.q_net_optimizers[i].load_state_dict(checkpoint["q_net_optimizers"][i])
        self.avg_net_optimizers[i].load_state_dict(checkpoint["avg_net_optimizers"][i])
      
      print(f"Checkpoint loaded. Resuming from iteration {start_iteration}")
      return start_iteration
    
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
                self.replay_buffer_capacity//self.num_actors,
                self.reservoir_buffer_capacity//self.num_actors,
                self.anticipatory_param,
                self.learning_rate,
                self.state_representation_size,
                self.hidden_layers_sizes,
                self.update_target_network_every,
                self.epsilon_start,
                self.epsilon_end,
                self.epsilon_decay_duration,
                self.batch_size//self.num_actors,
                self.min_buffer_size_to_learn,
                self.learn_every
                )
        )
    
    # ---------------------------------------------------------------------------
    # Federated Learning Methods
    # ---------------------------------------------------------------------------
    
    def collect_and_aggregate_gradients(self):
        """Collect gradients from all actors and aggregate them."""
        # print("Collecting gradients from all actors...")
        
        # Collect gradients from all actors for all players
        gradient_futures = []
        for actor in self.actors:
            for player_id in range(self.num_players):
                gradient_futures.append(actor.compute_gradients.remote(player_id))
        
        # Get all gradient results
        gradient_results = ray.get(gradient_futures)
        
        # Reorganize results by player
        player_gradients = {player_id: {'avg': [], 'q': [], 'losses': []} for player_id in range(self.num_players)}
        
        result_idx = 0
        for actor_idx in range(self.num_actors):
            for player_id in range(self.num_players):
                avg_grads, q_grads, sl_loss, rl_loss = gradient_results[result_idx]
                
                if avg_grads is not None:
                    player_gradients[player_id]['avg'].append(avg_grads)
                if q_grads is not None:
                    player_gradients[player_id]['q'].append(q_grads)
                    
                player_gradients[player_id]['losses'].append((sl_loss, rl_loss))
                result_idx += 1
        
        # Aggregate gradients for each player
        for player_id in range(self.num_players):
            if player_gradients[player_id]['avg']:
                avg_aggregated = self._average_gradients(player_gradients[player_id]['avg'])
                self._apply_gradients_to_avg_network(player_id, avg_aggregated)
                
            if player_gradients[player_id]['q']:
                q_aggregated = self._average_gradients(player_gradients[player_id]['q'])
                self._apply_gradients_to_q_network(player_id, q_aggregated)
        
        return player_gradients
    
    def _average_gradients(self, gradient_list):
        """Average gradients across multiple actors."""
        if not gradient_list:
            return None
            
        averaged_gradients = {}
        
        # Get parameter names from first gradient dict
        param_names = gradient_list[0].keys()
        
        for param_name in param_names:
            # Stack gradients from all actors for this parameter
            grad_tensors = [grads[param_name] for grads in gradient_list if param_name in grads]
            
            if grad_tensors:
                # Average the gradients
                stacked_grads = torch.stack(grad_tensors)
                averaged_gradients[param_name] = torch.mean(stacked_grads, dim=0)
        
        return averaged_gradients
    
    def _apply_gradients_to_avg_network(self, player_id, aggregated_gradients):
        """Apply aggregated gradients to the average policy network."""
        if aggregated_gradients is None:
            return
            
        # Zero gradients
        self.avg_net_optimizers[player_id].zero_grad()
        
        # Set the aggregated gradients
        for name, param in self.avg_networks[player_id].named_parameters():
            if name in aggregated_gradients:
                param.grad = aggregated_gradients[name]
        
        # Apply optimizer step
        self.avg_net_optimizers[player_id].step()
    
    def _apply_gradients_to_q_network(self, player_id, aggregated_gradients):
        """Apply aggregated gradients to the Q-network."""
        if aggregated_gradients is None:
            return
            
        # Zero gradients
        self.q_net_optimizers[player_id].zero_grad()
        
        # Set the aggregated gradients
        for name, param in self.q_networks[player_id].named_parameters():
            if name in aggregated_gradients:
                param.grad = aggregated_gradients[name]
        
        # Apply optimizer step
        self.q_net_optimizers[player_id].step()
    
    def distribute_updated_parameters(self):
        """Send updated network parameters to all actors."""
        # Get current network parameters
        q_network_params = [net.state_dict() for net in self.q_networks]
        avg_network_params = [net.state_dict() for net in self.avg_networks]
        
        # Send to all actors
        update_futures = []
        for actor in self.actors:
            update_futures.append(actor.update_networks.remote(q_network_params, avg_network_params))
        
        # Wait for all updates to complete
        ray.get(update_futures)
        # print("Updated parameters distributed to all actors")

    def start(self, resume_from_checkpoint=False, checkpoint_path=None):
        """Main training loop with federated learning and checkpointing."""
        # Create the remote NFSP actors.
        self.create_actors()
        
        # Handle checkpoint resuming
        start_iteration = 0
        if resume_from_checkpoint:
            if checkpoint_path:
                start_iteration = self.load_checkpoint(checkpoint_path)
            else:
                try:
                    start_iteration = self.load_checkpoint()  # Load latest
                except FileNotFoundError:
                    print("No checkpoint found, starting from scratch")
                    start_iteration = 0
        
        # Send initial network parameters to actors
        self.distribute_updated_parameters()
        
        print(f"Starting training from iteration {start_iteration + 1}/{self.num_iterations}")
        
        def _sample_episode_policy():
            # Sample an episode policy *independently for each player* so that
            # best-response / average-policy episodes are not perfectly
            # synchronised across all players.  This matches the design of the
            # reference implementation where each NFSP agent (one per player)
            # samples its own mode.
            modes = []
            for _ in range(self.num_players):
                if np.random.rand() < self.anticipatory_param:
                    modes.append(MODE.best_response)
                else:
                    modes.append(MODE.average_policy)
            return modes
        
        for iteration in range(start_iteration, self.num_iterations):
            # ------------------------------------------------------------------
            # 1) Generate trajectories (self-play)
            # ------------------------------------------------------------------
            modes = _sample_episode_policy()
            ray.get([actor.traverse_game.remote(iteration, modes) for actor in self.actors])
            
            # ------------------------------------------------------------------
            # 2) Federated learning: collect gradients, aggregate, and distribute
            # ------------------------------------------------------------------
            if (iteration + 1) % self.learn_every == 0:
                player_gradients = self.collect_and_aggregate_gradients()
                self.distribute_updated_parameters()
                
                # Print gradient statistics
                # for player_id in range(self.num_players):
                #     avg_count = len(player_gradients[player_id]['avg'])
                #     q_count = len(player_gradients[player_id]['q'])
                #     losses = player_gradients[player_id]['losses']
                #     print(f"Player {player_id}: {avg_count} avg gradients, {q_count} q gradients")
                #     print(f"Player {player_id} losses: {losses}")
            
            # ------------------------------------------------------------------
            # 3) Checkpointing every 100,000 iterations
            # ------------------------------------------------------------------
            if iteration % 100000 == 0:
                self.save_checkpoint(iteration)
    
            # ------------------------------------------------------------------
            # 4) Periodic evaluation 
            # ------------------------------------------------------------------
            if iteration % self.eval_every == 0:
                # Get training losses first
                loss_refs = [
                  actor.get_loss.remote(player)
                    for player in range(self.num_players)
                    for actor in self.actors
                ]
                losses = ray.get(loss_refs)
                
                # Organize losses by player for logging
                training_losses = {}
                for player in range(self.num_players):
                    training_losses[player] = losses[player]
                    print(f"Training losses (player {player}): {losses[player]}")
                
                # Evaluation - Create fresh evaluation policy to ensure we get current networks
                eval_ref = self.evaluator.comprehensive_evaluation.remote(
                    self.q_networks,
                    self.avg_networks,
                    num_head_to_head_episodes=1000,
                    iteration=iteration+1,
                    training_losses=training_losses
                )
                eval_results = ray.get(eval_ref)
                print(f"Iteration {iteration+1} - Evaluation Results:\n{eval_results}")

                print("----------------------------------------------------")
            
            if iteration % 1000 == 0:
              print(f"Completed iteration {iteration + 1}/{self.num_iterations}")
        
        # Save final checkpoint
        print("Training completed! Saving final checkpoint...")
        self.save_checkpoint(self.num_iterations)