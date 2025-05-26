import torch
def discretize(mu: torch.Tensor) -> torch.Tensor:
    """
    Discretize a batch of probability vectors into fixed-resolution distributions.

    Given a batch of “soft” probability vectors `mu` (shape `[B*T, n_actions]`),
    this function allocates exactly `policy_discretization` quanta (by default 32)
    across each vector’s entries in descending order of their original values.
    Each entry’s allocated quanta is the ceiling of its original weight-times-budget,
    but capped by the remaining budget as we sweep through actions from largest
    to smallest. Any leftover quanta are dumped into the top action. Finally, the
    integer counts are renormalized back into probabilities.

    Parameters
    ----------
    mu : torch.Tensor
        A batch of unbatched probability vectors, shape `(B*T, n_actions)`,
        where each row sums to 1 (or approximately so).

    Returns
    -------
    torch.Tensor
        A tensor of shape `(B*T, n_actions)` where each row is a probability
        vector that (a) sums exactly to 1, and (b) has values that are multiples
        of `1 / policy_discretization`.

    Example
    -------
    >>> mu = torch.tensor([[0.1, 0.2, 0.7],
    ...                    [0.33, 0.33, 0.34]])
    >>> discretize(mu)
    tensor([[0.0000, 0.1875, 0.8125],
            [0.3125, 0.3125, 0.3750]])
    """
    
    policy_discretization = 32
    mu_ = mu
    n_actions = mu_.shape[-1]
    roundup = torch.ceil(mu_ * policy_discretization)
    result = torch.zeros_like(mu_)
    order = torch.argsort(-mu_)  # Indices of descending order.
    weight_left = policy_discretization * torch.ones(mu_.shape[0])
    for i in range(n_actions):
        next_action = order[:,i]
        x = torch.minimum(roundup[torch.arange(next_action.size(0)),next_action], weight_left)
        addition_mask = torch.zeros_like(mu_)
        addition_mask[torch.arange(mu_.shape[0]),next_action] = 1
        weight_left_mask = weight_left >= 0
        result = result + (x.unsqueeze(1) * addition_mask * weight_left_mask.unsqueeze(1))
        weight_left -= x
        
    weight_left_next = weight_left
    result_next = result
    addition_mask_next = torch.zeros_like(mu_)
    addition_mask_next[torch.arange(mu_.shape[0]),order[:,0]] = 1
    weight_left_next_mask = weight_left_next > 0
    result_next = result_next + (weight_left_next.unsqueeze(1) * weight_left_next_mask.unsqueeze(1) * addition_mask_next)
    return result_next / policy_discretization

# Test the function
def test_discretize():
    # Test 1: Simple 2D tensor
    input_tensor = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    output = discretize(input_tensor)
    print("Test 1 - Input:", input_tensor)
    print("Test 1 - Output:", output)
    print("Test 1 - Output sum:", output.sum(dim=1))  # Should be close to 1
    
    # Test 2: Larger tensor
    batch_size = 3
    n_actions = 4
    random_tensor = torch.rand(batch_size, n_actions)
    # Normalize to make it a valid probability distribution
    random_tensor = random_tensor / random_tensor.sum(dim=1, keepdim=True)
    output = discretize(random_tensor)
    print("\nTest 2 - Input:", random_tensor)
    print("Test 2 - Output:", output)
    print("Test 2 - Output sum:", output.sum(dim=1))  # Should be close to 1
    
    # Test 3: Edge case - uniform distribution
    uniform_tensor = torch.ones(2, 3) / 3
    output = discretize(uniform_tensor)
    print("\nTest 3 - Input:", uniform_tensor)
    print("Test 3 - Output:", output)
    print("Test 3 - Output sum:", output.sum(dim=1))  # Should be close to 1

if __name__ == "__main__":
    test_discretize()