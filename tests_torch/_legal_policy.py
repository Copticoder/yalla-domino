import unittest
import torch
import numpy as np

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

class TestLegalPolicy(unittest.TestCase):

    def test_basic_case(self):
        logits = torch.tensor([[1.0, 2.0, 0.5, -1.0]])
        legal_actions = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
        policy = _legal_policy(logits, legal_actions)

        self.assertTrue(torch.allclose(policy.sum(dim=-1), torch.tensor([1.0])))
        self.assertEqual(policy[0, 3], 0.0) # Illegal action should have 0 probability
        self.assertTrue(policy[0, 0] > 0 and policy[0, 1] > 0 and policy[0, 2] > 0)
        # Check relative probabilities (exp(2-max) > exp(1-max) > exp(0.5-max))
        # max is 2.0. logits become [-1.0, 0.0, -1.5, -3.0] effectively for softmax part
        # exp_logits approx [0.367, 1.0, 0.223, 0.0]
        # sum approx 1.59
        # policy approx [0.23, 0.62, 0.14, 0.0]
        self.assertTrue(policy[0, 1] > policy[0, 0])
        self.assertTrue(policy[0, 0] > policy[0, 2])


    def test_all_actions_legal(self):
        logits = torch.tensor([[1.0, 0.0, -1.0]])
        legal_actions = torch.tensor([[1, 1, 1]], dtype=torch.bool)
        policy = _legal_policy(logits, legal_actions)

        self.assertTrue(torch.allclose(policy.sum(dim=-1), torch.tensor([1.0])))
        self.assertTrue(torch.all(policy > 0))
        # Max is 1.0. Logits become [0.0, -1.0, -2.0]
        # exp_logits approx [1.0, 0.367, 0.135]
        # sum approx 1.502
        # policy approx [0.665, 0.244, 0.090]
        self.assertTrue(policy[0,0] > policy[0,1] > policy[0,2])

    def test_one_action_legal(self):
        logits = torch.tensor([[1.0, 2.0, 0.5]])
        legal_actions = torch.tensor([[0, 1, 0]], dtype=torch.bool)
        policy = _legal_policy(logits, legal_actions)

        self.assertTrue(torch.allclose(policy.sum(dim=-1), torch.tensor([1.0])))
        self.assertEqual(policy[0, 0], 0.0)
        self.assertEqual(policy[0, 1], 1.0)
        self.assertEqual(policy[0, 2], 0.0)

    def test_no_actions_legal(self):
        logits = torch.tensor([[1.0, 2.0, 0.5]])
        legal_actions = torch.tensor([[0, 0, 0]], dtype=torch.bool)
        policy = _legal_policy(logits, legal_actions)
        # Expect NaN due to division by zero (sum of exp_logits will be 0)
        self.assertTrue(torch.all(torch.isnan(policy)))

    def test_negative_logits(self):
        logits = torch.tensor([[-1.0, -2.0, -0.5]])
        legal_actions = torch.tensor([[1, 1, 1]], dtype=torch.bool)
        policy = _legal_policy(logits, legal_actions)

        self.assertTrue(torch.allclose(policy.sum(dim=-1), torch.tensor([1.0])))
        self.assertTrue(torch.all(policy > 0))
        # Max is -0.5. Logits become [-0.5, -1.5, 0.0]
        # exp_logits approx [0.606, 0.223, 1.0]
        # sum approx 1.829
        # policy approx [0.331, 0.122, 0.547]
        self.assertTrue(policy[0,2] > policy[0,0] > policy[0,1])


    def test_batch_dimension(self):
        logits = torch.tensor([
            [1.0, 2.0, 0.5],
            [-1.0, 0.0, 1.0]
        ])
        legal_actions = torch.tensor([
            [1, 1, 0],
            [0, 1, 1]
        ], dtype=torch.bool)
        policy = _legal_policy(logits, legal_actions)

        self.assertTrue(torch.allclose(policy.sum(dim=-1), torch.tensor([1.0, 1.0])))
        self.assertEqual(policy[0, 2], 0.0)
        self.assertEqual(policy[1, 0], 0.0)
        self.assertTrue(policy[0,1] > policy[0,0]) # For first batch item
        self.assertTrue(policy[1,2] > policy[1,1]) # For second batch item

if __name__ == '__main__':
    unittest.main() 