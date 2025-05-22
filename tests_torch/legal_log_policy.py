import unittest
import torch
import numpy as np

def legal_log_policy(logits: torch.Tensor,
                     legal_actions: torch.Tensor) -> torch.Tensor:
  """Return the log of the policy on legal action, 0 on illegal action."""
  assert logits.shape == legal_actions.shape
  # logits_masked has illegal actions set to -inf.
  logits_masked = logits + torch.log(legal_actions)
  max_legal_logit = logits_masked.max(axis=-1, keepdims=True).values
  logits_masked = logits_masked - max_legal_logit
  # exp_logits_masked is 0 for illegal actions.
  exp_logits_masked = torch.exp(logits_masked)

  baseline = torch.log(torch.sum(exp_logits_masked, axis=-1, keepdims=True))
  # Subtract baseline from logits. We do not simply return
  #     logits_masked - baseline
  # because that has -inf for illegal actions, or
  #     legal_actions * (logits_masked - baseline)
  # because that leads to 0 * -inf == nan for illegal actions.
  log_policy = torch.multiply(legal_actions,
                            (logits - max_legal_logit - baseline))
  return log_policy

class TestLegalLogPolicy(unittest.TestCase):

    def assertTensorsClose(self, t1: torch.Tensor, t2: torch.Tensor, msg: str = None, rtol: float = 1e-5, atol: float = 1e-5):
        """
        Asserts that two tensors are close. Handles NaNs by checking if NaN positions match
        and then comparing non-NaN values.
        """
        # Ensure tensors are on the same device for comparison if needed, though usually test tensors are CPU.
        # self.assertEqual(t1.device, t2.device, f"Tensor devices do not match: {t1.device} vs {t2.device}")
        self.assertEqual(t1.dtype, t2.dtype, f"Tensor dtypes do not match: {t1.dtype} vs {t2.dtype}")

        nan_mask_t1 = torch.isnan(t1)
        nan_mask_t2 = torch.isnan(t2)
        self.assertTrue(torch.equal(nan_mask_t1, nan_mask_t2),
                        f"NaN mask mismatch. {msg}\n"
                        f"Tensor1 NaN mask:\n{nan_mask_t1}\nActual tensor1:\n{t1}\n"
                        f"Tensor2 NaN mask:\n{nan_mask_t2}\nExpected tensor2:\n{t2}")

        # Compare non-NaN parts
        # Create versions of tensors where NaNs are replaced by a common value (e.g., 0)
        # to make them comparable by assert_close, but only compare where original masks were False.
        # A simpler way: slice out non-NaNs.
        
        # If both are all NaNs and masks matched, this is fine.
        # If one is all NaNs and other is not, mask check would fail.
        # If both are empty (e.g. from empty non_nan selection), assert_close handles it.
        # We only apply assert_close to elements that are not NaN in t2 (and by extension, t1).
        if (~nan_mask_t2).any(): # If there are any non-NaN values in the expected tensor
            torch.testing.assert_close(t1[~nan_mask_t1], t2[~nan_mask_t2], rtol=rtol, atol=atol, msg=msg)
        # If t2 is all NaNs, and t1 is also all NaNs (checked by nan_mask_t1 == nan_mask_t2), then it's a pass.
        # No further check needed for non-NaN elements.

    def test_basic_case(self):
        logits = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32)
        legal_actions = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)
        # Manually calculate expected:
        # Legal logits: [1.0, 2.0]. Using torch.nn.functional.log_softmax for reference:
        # log_softmax(tensor([1.0, 2.0])) = tensor([-1.3133, -0.3133])
        expected = torch.tensor([-1.31326169, -0.31326169, 0.0], dtype=torch.float32)
        result = legal_log_policy(logits, legal_actions)
        self.assertTensorsClose(result, expected, "Basic case failed")

    def test_all_actions_legal(self):
        logits = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32)
        legal_actions = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        # Expected: log(softmax([1.0, 0.0, -1.0]))
        # log_softmax(tensor([1.0, 0.0, -1.0])) = tensor([-0.4076, -1.4076, -2.4076])
        expected = torch.tensor([-0.40760599, -1.40760601, -2.40760612], dtype=torch.float32)
        result = legal_log_policy(logits, legal_actions)
        self.assertTensorsClose(result, expected, "All actions legal failed")

    def test_one_action_legal(self):
        logits = torch.tensor([10.0, 20.0, 5.0], dtype=torch.float32)
        legal_actions = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        # Only action 1 is legal. Its probability is 1.0. Log(1.0) = 0.0.
        # Others are illegal, log_policy is 0.0.
        expected = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        result = legal_log_policy(logits, legal_actions)
        self.assertTensorsClose(result, expected, "One action legal failed")

    def test_all_actions_illegal(self):
        logits = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        legal_actions = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        # Current PyTorch code produces NaNs due to log(0) -> -inf, then -inf - (-inf) -> nan, then 0 * nan -> nan.
        expected = torch.tensor([float('nan'), float('nan'), float('nan')], dtype=torch.float32)
        result = legal_log_policy(logits, legal_actions)
        self.assertTensorsClose(result, expected, "All actions illegal failed. Expects NaNs due to 0*nan.")
        # Note: Docstring says "0 on illegal action". If this case should return [0,0,0],
        # the function needs modification. This test reflects current behavior.

    def test_batched_input_with_all_illegal_case(self):
        logits = torch.tensor([[1.0, 2.0, 0.5], [1.0, 2.0, 3.0]], dtype=torch.float32)
        legal_actions = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
        
        expected_row1 = torch.tensor([-1.31326169, -0.31326169, 0.0], dtype=torch.float32) # From test_basic_case
        expected_row2 = torch.tensor([float('nan'), float('nan'), float('nan')], dtype=torch.float32) # From test_all_actions_illegal
        expected = torch.stack([expected_row1, expected_row2])
        
        result = legal_log_policy(logits, legal_actions)
        self.assertTensorsClose(result, expected, "Batched input with all_illegal sub-case failed")

    def test_numerical_stability_large_logits(self):
        logits = torch.tensor([1000.0, 1001.0, 999.0], dtype=torch.float32)
        legal_actions = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)
        # Legal logits: [1000.0, 1001.0]. Max is 1001.0.
        # Normalized: [1000.0 - 1001.0, 1001.0 - 1001.0] = [-1.0, 0.0]
        # This is same as test_basic_case's effective logits [1.0, 2.0] -> normalized [-1.0, 0.0]
        # log_softmax(tensor([-1.0, 0.0])) = tensor([-1.3133, -0.3133])
        expected = torch.tensor([-1.31326169, -0.31326169, 0.0], dtype=torch.float32)
        result = legal_log_policy(logits, legal_actions)
        self.assertTensorsClose(result, expected, "Numerical stability with large logits failed")

    def test_logits_with_negative_infinity(self):
        logits = torch.tensor([1.0, -float('inf'), 0.5], dtype=torch.float32)
        legal_actions = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        # log_softmax(tensor([1.0, -inf, 0.5])) = tensor([-0.4741,   -inf, -0.9741])
        expected = torch.tensor([-0.47407299, -float('inf'), -0.97407299], dtype=torch.float32)
        result = legal_log_policy(logits, legal_actions)
        self.assertTensorsClose(result, expected, "Logits with -inf failed")

    def test_all_logits_equal_some_legal(self):
        logits = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        legal_actions = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
        # Legal logits: [1.0, 1.0] (for actions 0 and 2)
        # log_softmax(tensor([1.0, 1.0])) = tensor([-0.6931, -0.6931])
        expected = torch.tensor([-0.69314718, 0.0, -0.69314718], dtype=torch.float32)
        result = legal_log_policy(logits, legal_actions)
        self.assertTensorsClose(result, expected, "All logits equal, some legal failed")

if __name__ == '__main__':
    unittest.main()

