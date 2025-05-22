import torch
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage


batch_size = 32  # Example batch size
time_steps = 10  # Example number of time steps

# Create a torch tensor with shape (batch_size, time_steps, 8)
tensor = torch.zeros((batch_size, time_steps, 8))

print("h")


