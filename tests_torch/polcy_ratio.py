import torch
def policy_ratio(pi: torch.Tensor, mu: torch.Tensor, actions_oh: torch.Tensor,
                  valid: torch.Tensor) -> torch.Tensor:
  """Returns a ratio of policy pi/mu when selecting action a.

  By convention, this ratio is 1 on non valid states
  Args:
    pi: the policy of shape [..., A].
    mu: the sampling policy of shape [..., A].
    actions_oh: a one-hot encoding of the current actions of shape [..., A].
    valid: boolean tensor indicating valid states of shape [...,1].

  Returns:
    pi/mu on valid states and 1 otherwise. The shape is the same
    as pi, mu or actions_oh but without the last dimension A.
  """
  
  assert pi.shape == mu.shape == actions_oh.shape, "pi, mu, and actions_oh must have the same shape"
  assert valid.shape == actions_oh.shape[:-1], "valid must have the same shape as actions_oh except the last dimension"
  assert valid.dtype == torch.bool, "valid must be a boolean tensor"

  def _select_action_prob(pi):
    return (torch.sum(actions_oh * pi, axis=-1, keepdims=False) * valid +
            ~valid)

  pi_actions_prob = _select_action_prob(pi)
  mu_actions_prob = _select_action_prob(mu)
  return pi_actions_prob / mu_actions_prob  





def assert_tensors_almost_equal(t1, t2, test_name="", tol=1e-6):
    assert t1.shape == t2.shape, f"{test_name}: Shape mismatch. Expected {t2.shape}, Got {t1.shape}"
    assert torch.allclose(t1, t2, atol=tol, equal_nan=True), f"{test_name}: Expected \n{t2}\nGot \n{t1}"

def test_pr_basic_case():
    pi = torch.tensor([[0.6, 0.4]])
    mu = torch.tensor([[0.5, 0.5]])
    actions_oh = torch.tensor([[1.0, 0.0]])
    valid = torch.tensor([True])
    expected = torch.tensor([0.6 / 0.5])
    result = policy_ratio(pi, mu, actions_oh, valid)
    assert_tensors_almost_equal(result, expected, "test_pr_basic_case")

def test_pr_invalid_state():
    pi = torch.tensor([[0.6, 0.4]])
    mu = torch.tensor([[0.5, 0.5]])
    actions_oh = torch.tensor([[1.0, 0.0]])
    valid = torch.tensor([False]) # Invalid state
    expected = torch.tensor([1.0]) # Ratio should be 1
    result = policy_ratio(pi, mu, actions_oh, valid)
    assert_tensors_almost_equal(result, expected, "test_pr_invalid_state")

def test_pr_mu_zero_prob_valid_state():
    pi = torch.tensor([[0.6, 0.4]])
    mu = torch.tensor([[1.0, 0.0]]) # mu for chosen action is 0
    actions_oh = torch.tensor([[0.0, 1.0]]) # action 1 chosen
    valid = torch.tensor([True])
    # pi_actions_prob = 0.4 * 1 + 0 = 0.4
    # mu_actions_prob = 0.0 * 1 + 0 = 0.0
    # result = 0.4 / 0.0 = inf
    expected = torch.tensor([float('inf')])
    result = policy_ratio(pi, mu, actions_oh, valid)
    assert_tensors_almost_equal(result, expected, "test_pr_mu_zero_prob_valid_state")

def test_pr_pi_and_mu_zero_prob_valid_state():
    pi = torch.tensor([[1.0, 0.0]]) 
    mu = torch.tensor([[1.0, 0.0]]) 
    actions_oh = torch.tensor([[0.0, 1.0]]) # action 1 chosen, for which both pi and mu are 0
    valid = torch.tensor([True])
    # pi_actions_prob = 0.0 * 1 + 0 = 0.0
    # mu_actions_prob = 0.0 * 1 + 0 = 0.0
    # result = 0.0 / 0.0 = nan
    expected = torch.tensor([float('nan')])
    result = policy_ratio(pi, mu, actions_oh, valid)
    assert_tensors_almost_equal(result, expected, "test_pr_pi_and_mu_zero_prob_valid_state")


def test_pr_pi_zero_prob_valid_state():
    pi = torch.tensor([[1.0, 0.0]]) # pi for chosen action is 0
    mu = torch.tensor([[0.5, 0.5]])
    actions_oh = torch.tensor([[0.0, 1.0]]) # action 1 chosen
    valid = torch.tensor([True])
    # pi_actions_prob = 0.0 * 1 + 0 = 0.0
    # mu_actions_prob = 0.5 * 1 + 0 = 0.5
    # result = 0.0 / 0.5 = 0.0
    expected = torch.tensor([0.0])
    result = policy_ratio(pi, mu, actions_oh, valid)
    assert_tensors_almost_equal(result, expected, "test_pr_pi_zero_prob_valid_state")


def test_pr_batch_dimension():
    # Batch size 2
    pi = torch.tensor([[0.6, 0.4], [0.1, 0.9]])
    mu = torch.tensor([[0.5, 0.5], [0.8, 0.2]])
    actions_oh = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    valid = torch.tensor([True, True])
    expected_b0 = 0.6 / 0.5
    expected_b1 = 0.9 / 0.2
    expected = torch.tensor([expected_b0, expected_b1])
    result = policy_ratio(pi, mu, actions_oh, valid)
    assert_tensors_almost_equal(result, expected, "test_pr_batch_dimension")

def test_pr_batch_with_invalid():
    # Batch size 3
    pi = torch.tensor([[0.6, 0.4], [0.1, 0.9], [0.7, 0.3]])
    mu = torch.tensor([[0.5, 0.5], [0.8, 0.2], [0.2, 0.8]])
    actions_oh = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    valid = torch.tensor([True, False, True]) # Second one is invalid
    
    # Batch 0: (0.6 * 1) / (0.5 * 1) = 1.2
    # Batch 1: invalid, so 1.0
    # Batch 2: (0.7 * 1) / (0.2 * 1) = 3.5
    expected = torch.tensor([0.6/0.5, 1.0, 0.7/0.2])
    result = policy_ratio(pi, mu, actions_oh, valid)
    assert_tensors_almost_equal(result, expected, "test_pr_batch_with_invalid")

def test_pr_multiple_dims():
    # Shape (T, B, A) for policies, (T, B) for valid
    # T=2, B=2, A=2
    pi = torch.tensor([
        [[0.6, 0.4], [0.1, 0.9]], # T0
        [[0.3, 0.7], [0.5, 0.5]]  # T1
    ])
    mu = torch.tensor([
        [[0.5, 0.5], [0.8, 0.2]], # T0
        [[0.2, 0.8], [0.4, 0.6]]  # T1
    ])
    actions_oh = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]], # T0, A0 for B0; A1 for B1
        [[0.0, 1.0], [1.0, 0.0]]  # T1, A1 for B0; A0 for B1
    ])
    valid = torch.tensor([
        [True, True], # T0
        [True, False]  # T1, B1 is invalid
    ])

    # T0, B0: pi=0.6, mu=0.5 => 1.2
    # T0, B1: pi=0.9, mu=0.2 => 4.5
    # T1, B0: pi=0.7, mu=0.8 => 0.875
    # T1, B1: invalid => 1.0
    expected = torch.tensor([
        [0.6/0.5, 0.9/0.2],
        [0.7/0.8, 1.0]
    ])
    result = policy_ratio(pi, mu, actions_oh, valid)
    assert_tensors_almost_equal(result, expected, "test_pr_multiple_dims")

def test_pr_float64():
    pi = torch.tensor([[0.6, 0.4]], dtype=torch.float64)
    mu = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    actions_oh = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    valid = torch.tensor([True])
    expected = torch.tensor([0.6 / 0.5], dtype=torch.float64)
    result = policy_ratio(pi, mu, actions_oh, valid)
    assert_tensors_almost_equal(result, expected, "test_pr_float64")

if __name__ == "__main__":
    test_pr_basic_case()
    test_pr_invalid_state()
    test_pr_mu_zero_prob_valid_state()
    test_pr_pi_and_mu_zero_prob_valid_state()
    test_pr_pi_zero_prob_valid_state()
    test_pr_batch_dimension()
    test_pr_batch_with_invalid()
    test_pr_multiple_dims()
    test_pr_float64()

    print("All policy_ratio tests passed (if no assertion errors)!")

