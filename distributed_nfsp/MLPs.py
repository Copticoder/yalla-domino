import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy import stats
from typing import Sequence, List

class SonnetLinear(nn.Module):
  """A Sonnet linear module.

  Always includes biases and only supports ReLU activations.
  """

  def __init__(self, in_size, out_size, activate_relu=True, use_layer_norm=False):
    """Creates a Sonnet linear layer.

    Args:
      in_size: (int) number of inputs
      out_size: (int) number of outputs
      activate_relu: (bool) whether to include a ReLU activation layer
      use_layer_norm: (bool) whether to apply layer normalization
    """
    super(SonnetLinear, self).__init__()
    self._activate_relu = activate_relu
    self._use_layer_norm = use_layer_norm
    stddev = 1.0 / math.sqrt(in_size)
    mean = 0
    lower = (-2 * stddev - mean) / stddev
    upper = (2 * stddev - mean) / stddev
    # Weight initialization inspired by Sonnet's Linear layer,
    # which cites https://arxiv.org/abs/1502.03167v3
    # pytorch default: initialized from
    # uniform(-sqrt(1/in_features), sqrt(1/in_features))
    self._weight = nn.Parameter(
        torch.Tensor(
            stats.truncnorm.rvs(
                lower, upper, loc=mean, scale=stddev, size=[out_size,
                                                            in_size])))
    self._bias = nn.Parameter(torch.zeros([out_size]))
    
    if self._use_layer_norm:
      self._layer_norm = nn.LayerNorm(out_size)

  def forward(self, tensor):
    y = F.linear(tensor, self._weight, self._bias)
    if self._use_layer_norm:
      y = self._layer_norm(y)
    return F.relu(y) if self._activate_relu else y


class BR_MLP(nn.Module):
  """A simple network built from nn.linear layers."""

  def __init__(self,
               input_size,
               hidden_sizes,
               output_size,
               activate_final=False,
               use_layer_norm=False):
    """Create the MLP.

    Args:
      input_size: (int) number of inputs
      hidden_sizes: (list) sizes (number of units) of each hidden layer
      output_size: (int) number of outputs
      activate_final: (bool) should final layer should include a ReLU
      use_layer_norm: (bool) whether to apply layer normalization to all layers
    """

    super(BR_MLP, self).__init__()
    self._layers = []
    # Hidden layers
    for size in hidden_sizes:
      self._layers.append(SonnetLinear(in_size=input_size, out_size=size, use_layer_norm=use_layer_norm))
      input_size = size
    # Output layer
    self._layers.append(
        SonnetLinear(
            in_size=input_size,
            out_size=output_size,
            activate_relu=activate_final,
            use_layer_norm=use_layer_norm))

    self.model = nn.ModuleList(self._layers)

  def forward(self, x):
    for layer in self.model:
      x = layer(x)
    return x

class AVG_MLP(nn.Module):
  """Simple MLP identical to the one used in the DQN PyTorch agent."""

  def __init__(self, in_size: int, hidden_sizes: Sequence[int], out_size: int, use_layer_norm: bool = False):
    super().__init__()
    self._use_layer_norm = use_layer_norm
    sizes = list(hidden_sizes) + [out_size]
    layers: List[nn.Module] = []
    for i, hs in enumerate(sizes[:-1]):
      layers.append(nn.Linear(in_size, hs))
      if self._use_layer_norm:
        layers.append(nn.LayerNorm(hs))
      layers.append(nn.ReLU())
      in_size = hs
    layers.append(nn.Linear(in_size, sizes[-1]))  # last layer – no activation
    if self._use_layer_norm:
      layers.append(nn.LayerNorm(sizes[-1]))
    self._model = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
    return self._model(x)