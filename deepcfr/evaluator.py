from open_spiel.python import policy as policy_module
from open_spiel.python.algorithms import exploitability
import ray
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import os
class Evaluator(policy_module.Policy):
    def __init__(self, game, policy_network, evaluation_interval, policy_network_train_steps, num_actors, learning_rate, run, strategy_learner):
        self.game = game
        self.policy_network = policy_network
        self.evaluation_interval = evaluation_interval
        self.policy_network_train_steps = policy_network_train_steps
        self.num_actors = num_actors
        self.learning_rate = learning_rate
        self._policy_network = policy_network
        self._optimizer_policy = torch.optim.Adam(self._policy_network.parameters(), lr=self.learning_rate)
        self.wandb_run = run
        self._policy_sm = nn.Softmax(dim=-1)
        self.strategy_learner = strategy_learner
    
    def evaluate(self, num_unique_info_states):
        # Train the global strategy network through the dedicated learner.
        strategy_loss = ray.get(self.strategy_learner.learn.remote())

        # Pull latest weights from the strategy learner.
        state_dict = ray.get(self.strategy_learner.get_policy_network_state.remote())
        self._policy_network.load_state_dict(state_dict)
        policy_losses = [strategy_loss]

        player_0_returns, player_1_returns = self.evaluate_agent()
        print(f"Player 0 returns: {player_0_returns}, Player 1 returns: {player_1_returns}")
        conv = None
        if self.game.get_type().short_name != "python_block_dominoes":
            policy = policy_module.tabular_policy_from_callable(self.game, self.action_probabilities)
            conv = exploitability.nash_conv(self.game, policy)
            print("Deep CFR - NashConv:", conv)  
        # Reinitialize the optimiser so future training steps start fresh.
        self._optimizer_policy = torch.optim.Adam(self._policy_network.parameters(), lr=self.learning_rate)
        if self.wandb_run:
            self.wandb_run.log({"nash_conv": conv, "visited_unique_info_states": num_unique_info_states, "player_0_running_score": player_0_returns-player_1_returns, "strategy_loss": strategy_loss})
        self.save_policy_network("./networks/policy_network.pth")
        return policy_losses
    
    def train_policy_network(self):
        """Train the policy network via gradient aggregation across all actors."""
        policy_losses = []

        for _ in tqdm(range(self.policy_network_train_steps), desc="Training policy network"):
            # Collect gradients and losses from ALL actors of BOTH players.
            remote_tasks = []
            for player in range(self.game.num_players()):
                actors = [ray.get_actor(f"actor_{player}_{i}", namespace="deep_cfr")
                          for i in range(self.num_actors // self.game.num_players())]
                remote_tasks += [actor.policy_network_step.remote(self._policy_network) for actor in actors]

            # Fetch results. Some actors might return `None` if their strategy
            # buffer is still too small – filter those out.
            results = [res for res in ray.get(remote_tasks) if res is not None]
            if not results:
                # Not enough data yet – skip this optimisation step.
                continue

            gradients_list, losses_list = zip(*results)
            policy_losses.append(float(np.mean(losses_list)))

            # Average gradients across all actors.
            averaged_grads = []
            for grads_per_param in zip(*gradients_list):
                averaged_grads.append(torch.stack(grads_per_param).mean(dim=0))

            # Apply gradients to the local policy network.
            self._optimizer_policy.zero_grad()
            for param, grad in zip(self._policy_network.parameters(), averaged_grads):
                param.grad = grad
            self._optimizer_policy.step()
        return policy_losses
    
    def evaluate_agent(self, num_episodes=100):
        """evaluate the agent on the game against a random agent"""
        player_0_returns = np.array([])
        player_1_returns = np.array([])
        for _ in tqdm(range(num_episodes), desc="Evaluating agent"):
            state = self.game.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    chance_outcome, chance_proba = zip(*state.chance_outcomes())
                    action = np.random.choice(chance_outcome, p=chance_proba)
                elif state.current_player() == 0:
                    probs_dict = self.action_probabilities(state)
                    total_p = sum(probs_dict.values())
                    if total_p <= 0.0:
                        # Shouldn't happen due to earlier guard, but fallback
                        legal_actions = list(probs_dict.keys())
                        action = np.random.choice(legal_actions)
                    else:
                        # Ensure exact normalisation
                        probs = np.array(list(probs_dict.values()), dtype=np.float64)
                        probs /= probs.sum()
                        action = np.random.choice(list(probs_dict.keys()), p=probs)
                else:
                    # take random action
                    action = np.random.choice(state.legal_actions())
                state = state.child(action)
            player_0_returns = np.append(player_0_returns, state.returns()[0])
            player_1_returns = np.append(player_1_returns, state.returns()[1])
        return np.sum(player_0_returns) / num_episodes, np.sum(player_1_returns) / num_episodes
    
    def calculate_exploitability(self):
        """Compute exploitability of the policy."""
        policy = policy_module.tabular_policy_from_callable(self.game, self.action_probabilities)
        return exploitability.nash_conv(self.game, policy)
    
   
    def action_probabilities(self, state):
        """Computes action probabilities for the current player in state."""
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        info_state_vector = np.array(state.information_state_tensor())
        if len(info_state_vector.shape) == 1:
            info_state_vector = np.expand_dims(info_state_vector, axis=0)
        with torch.no_grad():
            logits = self._policy_network(torch.FloatTensor(info_state_vector))
            probs = self._policy_sm(logits).numpy()[0]

        # Filter to legal actions and renormalise to guard against rounding
        action_probs = {action: float(probs[action]) for action in legal_actions}
        total = sum(action_probs.values())
        if total <= 0.0:
            # Fall back to uniform distribution if network assigns zero mass.
            uniform = 1.0 / len(legal_actions)
            action_probs = {a: uniform for a in legal_actions}
        else:
            action_probs = {a: p / total for a, p in action_probs.items()}
        return action_probs

    def save_policy_network(self, path):
        # make sure parent directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_network.state_dict(), path)

    def load_policy_network(self, path):
        self.policy_network.load_state_dict(torch.load(path))