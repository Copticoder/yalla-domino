import numpy as np
import torch
import torch.nn.functional as F
import ray
from typing import Dict, List
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability


class NFSPPolicy(policy.Policy):
    """Policy wrapper for NFSP networks that can be used with OpenSpiel evaluators."""
    
    def __init__(self, game, player_id: int, network: torch.nn.Module, num_actions: int, mode: str = "average"):
        """Initialize NFSP policy.
        
        Args:
            game: OpenSpiel game
            player_id: Player ID (0 or 1)
            network: The neural network (either Q-network or average policy network)
            num_actions: Number of possible actions
            mode: Either "average" for average policy or "best_response" for Q-network
        """
        super().__init__(game, [player_id])
        self._network = network
        self._num_actions = num_actions
        self._mode = mode
        self._player_id = player_id
        
    def action_probabilities(self, state, player_id=None):
        """Get action probabilities for the current state."""
        if state.is_terminal() or state.current_player() != self._player_id:
            return {}
            
        info_state = torch.Tensor(state.information_state_tensor()).unsqueeze(0)
        legal_actions = state.legal_actions()
        
        with torch.no_grad():
            if self._mode == "average":
                # For average policy, use softmax over network outputs
                logits = self._network(info_state)
                action_probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            else:  # best_response mode
                # For Q-network, use greedy action selection
                q_values = self._network(info_state).squeeze(0).cpu().numpy()
                action_probs = np.zeros(self._num_actions)
                best_action = legal_actions[np.argmax(q_values[legal_actions])]
                action_probs[best_action] = 1.0
        
        # Ensure only legal actions have non-zero probabilities
        probs = {}
        for action in legal_actions:
            probs[action] = action_probs[action]
            
        # Normalize probabilities for legal actions only
        total_prob = sum(probs.values())
        if total_prob > 0:
            for action in probs:
                probs[action] /= total_prob
        else:
            # Fallback to uniform distribution over legal actions
            for action in legal_actions:
                probs[action] = 1.0 / len(legal_actions)
                
        return probs


class RandomPolicy(policy.Policy):
    """Random policy for head-to-head evaluation."""
    
    def __init__(self, game, player_id: int):
        super().__init__(game, [player_id])
        self._player_id = player_id
        
    def action_probabilities(self, state, player_id=None):
        """Return uniform probabilities over legal actions."""
        if state.is_terminal() or state.current_player() != self._player_id:
            return {}
            
        legal_actions = state.legal_actions()
        prob = 1.0 / len(legal_actions)
        return {action: prob for action in legal_actions}


@ray.remote(num_cpus=1, namespace="nfsp")
class Evaluator:
    """Evaluator actor for NFSP players."""
    
    def __init__(self, game, num_players: int = 2, num_actions: int = None):
        """Initialize the evaluator.
        
        Args:
            game: OpenSpiel game instance
            num_players: Number of players (default 2)
            num_actions: Number of possible actions
        """
        self._game = game
        self._num_players = num_players
        self._num_actions = num_actions if num_actions else game.num_distinct_actions()
        
    def evaluate_head_to_head(self, 
                            q_networks: List[torch.nn.Module], 
                            avg_networks: List[torch.nn.Module],
                            num_episodes: int = 1000,
                            use_average_policy: bool = True) -> Dict[str, float]:
        """Evaluate NFSP players head-to-head against random agents.
        
        Args:
            q_networks: List of Q-networks for each player
            avg_networks: List of average policy networks for each player  
            num_episodes: Number of episodes to evaluate
            use_average_policy: Whether to use average policy (True) or best response (False)
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            'nfsp_wins': 0,
            'random_wins': 0,
            'draws': 0,
            'nfsp_avg_reward': 0.0,
            'random_avg_reward': 0.0
        }
        
        networks = avg_networks if use_average_policy else q_networks
        mode = "average" if use_average_policy else "best_response"
        
        # Create policies
        nfsp_policies = [
            NFSPPolicy(self._game, player_id, networks[player_id], self._num_actions, mode)
            for player_id in range(self._num_players)
        ]
        
        random_policies = [
            RandomPolicy(self._game, player_id)
            for player_id in range(self._num_players)
        ]
        
        total_nfsp_reward = 0.0
        total_random_reward = 0.0
        
        for episode in range(num_episodes):
            # Alternate which player is NFSP vs random
            nfsp_player = episode % self._num_players
            
            # Create joint policy (NFSP vs Random)
            policies = [None] * self._num_players
            policies[nfsp_player] = nfsp_policies[nfsp_player]
            for i in range(self._num_players):
                if i != nfsp_player:
                    policies[i] = random_policies[i]
            
            # Run episode
            state = self._game.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    # Handle chance nodes
                    legal_actions = state.legal_actions()
                    action = np.random.choice(legal_actions)
                else:
                    # Get action from appropriate policy
                    current_player = state.current_player()
                    action_probs = policies[current_player].action_probabilities(state)
                    actions = list(action_probs.keys())
                    probs = np.array(list(action_probs.values()), dtype=np.float64)
                    probs = np.clip(probs, 0.0, None)        # 1) ensure non-negative
                    total = probs.sum()
                    if total == 0.0:                         # 2) fallback to uniform
                        probs = np.ones_like(probs) / len(probs)
                    else:                                    # 3) renormalise
                        probs = probs / total
                    action = np.random.choice(actions, p=probs)
                state = state.child(action)
            
            # Get final rewards
            returns = state.returns()
            nfsp_reward = returns[nfsp_player]
            random_reward = sum(returns[i] for i in range(self._num_players) if i != nfsp_player)
            
            total_nfsp_reward += nfsp_reward
            total_random_reward += random_reward
            
            # Determine winner (for zero-sum games)
            if nfsp_reward > random_reward:
                results['nfsp_wins'] += 1
            elif random_reward > nfsp_reward:
                results['random_wins'] += 1
            else:
                results['draws'] += 1
        
        # Calculate averages
        results['nfsp_avg_reward'] = total_nfsp_reward / num_episodes
        results['random_avg_reward'] = total_random_reward / num_episodes
        results['nfsp_win_rate'] = results['nfsp_wins'] / num_episodes
        
        return results
    
    def calculate_exploitability(self, 
                                avg_networks: List[torch.nn.Module]) -> float:
        """Calculate Nash-conv (exploitability) of average policy using OpenSpiel.
        
        Args:
            avg_networks: List of average policy networks for each player
            
        Returns:
            Exploitability value
        """
        # Create joint policy from average networks
        policies = [
            NFSPPolicy(self._game, player_id, avg_networks[player_id], self._num_actions, "average")
            for player_id in range(self._num_players)
        ]
        
        # Create joint policy for exploitability calculation
        joint_policy = JointPolicy(self._game, policies)
        
        # Calculate exploitability
        try:
            expl = exploitability.exploitability(self._game, joint_policy)
            return expl
        except Exception as e:
            print(f"Error calculating exploitability: {e}")
            return float('inf')
    
    def calculate_nash_conv(self, 
                           avg_networks: List[torch.nn.Module]) -> float:
        """Calculate Nash convergence metric.
        
        This is an alias for calculate_exploitability for backward compatibility.
        """
        return self.calculate_exploitability(avg_networks)
    
    def comprehensive_evaluation(self, 
                               q_networks: List[torch.nn.Module],
                               avg_networks: List[torch.nn.Module],
                               num_head_to_head_episodes: int = 1000) -> Dict[str, any]:
        """Run comprehensive evaluation including both head-to-head and exploitability.
        
        Args:
            q_networks: List of Q-networks for each player
            avg_networks: List of average policy networks for each player
            num_head_to_head_episodes: Number of episodes for head-to-head evaluation
            
        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}
        
        # Head-to-head evaluation with average policy
        print("Evaluating average policy head-to-head...")
        h2h_avg = self.evaluate_head_to_head(
            q_networks, avg_networks, num_head_to_head_episodes, use_average_policy=True
        )
        results['head_to_head_average'] = h2h_avg
        
        # Head-to-head evaluation with best response
        print("Evaluating best response head-to-head...")
        h2h_br = self.evaluate_head_to_head(
            q_networks, avg_networks, num_head_to_head_episodes, use_average_policy=False
        )
        results['head_to_head_best_response'] = h2h_br
        
        # Exploitability calculation
        print("Calculating exploitability...")
        exploitability_score = self.calculate_exploitability(avg_networks)
        results['exploitability'] = exploitability_score
        
        # Summary metrics
        results['summary'] = {
            'avg_policy_win_rate': h2h_avg['nfsp_win_rate'],
            'br_policy_win_rate': h2h_br['nfsp_win_rate'],
            'exploitability': exploitability_score,
            'nash_conv': exploitability_score  # Same as exploitability
        }
        
        return results
    
    def update_networks(self, 
                       q_networks_state_dicts: List[Dict], 
                       avg_networks_state_dicts: List[Dict],
                       hidden_layers_sizes: List[int],
                       state_representation_size: int):
        """Update evaluator with new network parameters from parameter server.
        
        Args:
            q_networks_state_dicts: State dictionaries for Q-networks
            avg_networks_state_dicts: State dictionaries for average networks
            hidden_layers_sizes: Hidden layer sizes for network reconstruction
            state_representation_size: Input size for networks
        """
        # Recreate networks with the provided state dictionaries
        self._q_networks = []
        self._avg_networks = []
        
        for player_id in range(self._num_players):
            # Create Q-network and load state
            q_net = MLP(state_representation_size, hidden_layers_sizes, self._num_actions)
            q_net.load_state_dict(q_networks_state_dicts[player_id])
            q_net.eval()  # Set to evaluation mode
            self._q_networks.append(q_net)
            
            # Create average network and load state
            avg_net = MLP(state_representation_size, hidden_layers_sizes, self._num_actions)
            avg_net.load_state_dict(avg_networks_state_dicts[player_id])
            avg_net.eval()  # Set to evaluation mode
            self._avg_networks.append(avg_net)


class JointPolicy(policy.Policy):
    """Joint policy that combines individual player policies for exploitability calculation."""
    
    def __init__(self, game, individual_policies: List[policy.Policy]):
        """Initialize joint policy.
        
        Args:
            game: OpenSpiel game
            individual_policies: List of individual policies for each player
        """
        all_players = []
        for pol in individual_policies:
            all_players.extend(pol.player_ids)
        super().__init__(game, all_players)
        self._policies = individual_policies
        
    def action_probabilities(self, state, player_id=None):
        """Get action probabilities from the appropriate individual policy."""
        if state.is_terminal():
            return {}
            
        current_player = state.current_player()
        if current_player >= len(self._policies):
            return {}
            
        return self._policies[current_player].action_probabilities(state, player_id)