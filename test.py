# discounted_return_vmap.py
#
# Compute discounted return-to-go for a batch of episodes.
# Each R_t depends on future rewards in the *same* episode,
# but episodes are independent, so we vmap over that dimension.

import time
import torch
from torch.func import vmap           # functorch.vmap if < PyTorch 2.0

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------
# 1. Single-episode discounted return (depends on future time steps)
# --------------------------------------------------------------------
def returns_one_episode(rewards, gamma: float):
    """
    rewards: 1-D tensor [T]
    returns: 1-D tensor [T] with R_t = r_t + γ R_{t+1}
    """
    T = rewards.size(0)
    out = torch.empty_like(rewards)
    G   = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(T)):       # causal dependency
        G = rewards[t] + gamma * G
        out[t] = G
    return out

# --------------------------------------------------------------------
# 2. Fake batch of episodes
# --------------------------------------------------------------------
B, T   = 8192, 512                    # batch size & episode length
gamma  = 0.99
rewards = torch.randn(B, T, device=device)

# --------------------------------------------------------------------
# 3a. Plain Python loop over episodes
# --------------------------------------------------------------------
t0 = time.perf_counter()
returns_loop = torch.stack([
    returns_one_episode(rewards[i], gamma) for i in range(B)
])
time_loop = time.perf_counter() - t0

# --------------------------------------------------------------------
# 3b. Vectorised version with vmap (eliminates outer loop)
# --------------------------------------------------------------------
returns_vmap_fn = vmap(returns_one_episode, in_dims=(0, None))
t0 = time.perf_counter()
returns_vmap = returns_vmap_fn(rewards, gamma)
time_vmap = time.perf_counter() - t0

# --------------------------------------------------------------------
# 4. Validate and report
# --------------------------------------------------------------------
print(f"Loop time : {time_loop:7.3f} s")
print(f"vmap time : {time_vmap:7.3f} s  ({time_loop / time_vmap:,.1f}× faster)")
print("Results identical:", torch.allclose(returns_loop, returns_vmap))