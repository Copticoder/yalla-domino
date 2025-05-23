from torch import vmap
import torch
def discretize_single(mu: torch.Tensor) -> torch.Tensor:
    policy_discretization = 32
    """A version of self._discretize but for the unbatched data."""
    mu_ = mu
    n_actions = mu_.shape[-1]
    roundup = torch.ceil(mu_ * policy_discretization)
    result = torch.zeros_like(mu_)
    order = torch.argsort(-mu_)  # Indices of descending order.
    weight_left = policy_discretization
    for i in range(roundup.int().item()):
        next_action = order[i]
        x = torch.minimum(roundup[next_action], weight_left)
        result = torch.where(weight_left >= 0, result.index_add(0, next_action.unsqueeze(0), x.unsqueeze(0)), result)
        weight_left -= x
    weight_left_next = weight_left
    result_next = result
    result_next = torch.where(weight_left_next > 0,
                            result_next.index_add(0, order[0].unsqueeze(0), weight_left_next.unsqueeze(0)),
                            result_next)
    if len(mu.shape) == 2:
      result_next = torch.expand_dims(result_next, axis=0)
    return result_next / policy_discretization
        
    
batch_size = 10
time_steps = 5
action_size = 10
tensor = torch.randn(batch_size, time_steps, action_size)

# First flatten over batch and time
tensor_reshaped = torch.abs(tensor.view(-1, action_size))  # shape (B*T, A)
tensor_reshaped = tensor_reshaped/tensor_reshaped.sum(dim=1,keepdim=True)
# Vectorize your function
# v_fn = vmap(discretize_single, in_dims=0, out_dims=0)

# Apply
output = discretize_single(tensor_reshaped[0,:])

# Reshape back
output = output.view(batch_size, time_steps, -1)


