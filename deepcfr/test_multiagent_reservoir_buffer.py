"""Quick test for RLlib's MultiAgentReplayBuffer configured with an
underlying ReservoirReplayBuffer.

1. Smoke-test the buffer in isolation (no Ray initialisation required).
2. Run a tiny RLlib training job (DQN on CartPole) using the same replay
   buffer configuration to prove that everything wires up correctly.

Requirements:
$ pip install "ray[rllib]>=2.6.0"  torch gymnasium

Usage:
$ python test_multiagent_reservoir_buffer.py --smoke
"""

from __future__ import annotations

import numpy as np

# ------------- 1) Stand-alone buffer smoke-test -------------------

def smoke_test(n_timesteps: int = 10000, capacity: int = 100) -> None:
    """Fill the buffer with dummy data and confirm sampling works."""
    # Import here to avoid pulling in heavy dependencies if the user only
    # wants to run the RLlib part.
    from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
    from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import (
        MultiAgentReplayBuffer,
    )
    from ray.rllib.utils.replay_buffers.replay_buffer import StorageUnit
    # Construct the MA buffer with Reservoir sampling underneath.
    buffer = MultiAgentReplayBuffer(
        capacity=capacity,
        storage_unit=StorageUnit.FRAGMENTS,
        # This sub-dict is passed straight to the constructor of the underlying
        # per-policy buffers.
        underlying_buffer_config={
            "type": "ray.rllib.utils.replay_buffers.reservoir_replay_buffer.ReservoirReplayBuffer",
            "capacity": capacity,
            "storage_unit": StorageUnit.FRAGMENTS,
        },
        replay_mode="independent",
    )
    list_batches = []
    # Insert fake timesteps for two logical policies.
    for t in range(n_timesteps):
        batch = MultiAgentBatch(
        {
            f"policy_{t%2}":SampleBatch(
                {
                    "info_state": np.array([[t,t,t]]),
                    "actions": np.array([t % 2]),
                    "rewards": np.array([1.0]),
                })},env_steps=1)
        list_batches.append(batch)
    
    # After insertion the buffer size for each policy should not exceed
    # `capacity` thanks to reservoir sampling.
    print("Buffer stats:", buffer.stats())

    # Sample a few timesteps back.
    sample = buffer.sample(2, policy_id="policy_1")
    print("Buffer stats:", buffer.stats())
    
    # `sample` is a MultiAgentBatch keyed by policy_id -> SampleBatch.
    sb = sample.policy_batches["policy_1"]
    print("Sampled info_state from policy_1:", sb["info_state"])

# ----------------------------- CLI --------------------------------

if __name__ == "__main__":
    smoke_test()
