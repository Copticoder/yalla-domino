# Yalla Domino – Distributed NFSP Training with OpenSpiel

A research-grade implementation of **Neural Fictitious Self-Play (NFSP)** for the game *Draw Dominoes* built on [OpenSpiel](https://github.com/deepmind/open_spiel), **PyTorch**, and **Ray**.  The system scales from a laptop to a multi-node cluster, provides automatic checkpointing, and logs rich metrics to [Weights & Biases](https://wandb.ai/).

---

## Key Features

* **Distributed training** – parallel actors implemented with Ray for fast self-play data generation.
* **Federated parameter updates** – gradients are aggregated centrally and broadcast back to actors each step.
* **Checkpointing & resume** – automatic saving every 10 k iterations plus manual resume from any snapshot.
* **Comprehensive evaluation** – head-to-head win-rate, exploitability (Nash-conv) and custom metrics.
* **First-class experiment tracking** – optional Weights & Biases integration with one-line opt-in.
* **Pure Python** – only a tiny C++/PyBind11 component from OpenSpiel is compiled.

---

## Repository Layout

```
.
├── runner.py            # Entrypoint – parses flags & launches the Ray cluster
├── Learner.py           # Parameter server / federated learner
├── NFSPActor.py         # Remote self-play actors (best-response & average policy)
├── Evaluator.py         # Detached actor for periodic evaluation & WandB logging
├── DQN.py               # Vanilla DQN agent (building block of NFSP)
├── MLPs.py              # Lightweight MLP implementations (BR & AVG nets)
├── checkpoints/         # Auto-generated – training snapshots (.pkl)
├── open_spiel/          # Vendored fork of DeepMind OpenSpiel (includes Dominoes)
└── …                    # Utilities, tests, third-party libs
```

---

## Quick Start

### 1. source the venv

```bash
$ source venv/bin/activate
```

### 2. Install deps
NOTE: It's already installed in after unzipping, skip this step if you're using the provided venv.
1. System:
   * `cmake`, `gcc` ≥ 9, `make`, `libopenmpi-dev` (Ubuntu: `sudo apt install build-essential cmake libopenmpi-dev`)
2. Python (CPU – adjust torch version for CUDA):

```bash
(venv) $ pip install torch==2.0.1 \
                  ray[default]==2.9.3 \
                  absl-py==1.4.0 \
                  wandb==0.16.6 \
                  numpy scipy tqdm

# Build & install OpenSpiel (takes a few minutes)
(venv) $ cd open_spiel && ./install.sh && pip install -e . && cd ..
```

### 3. Launch training

```bash
(venv) $ python runner.py 
```

Flags are defined in `runner.py` via **absl** 

Training checkpoints are written to `checkpoints/checkpoint_iter_<N>.pkl` and a symlink `checkpoint_latest.pkl`.

### 4. Resume from checkpoint
edit the `runner.py` file to set the checkpoint path and the number of iterations to resume from.
```bash
(venv) $ python runner.py 
```

### 5. Monitor metrics

If Weights & Biases is enabled, visit the run URL printed in the logs to view live:

* win-rate vs random baseline (average & best-response)
* exploitability / Nash-conv
* per-player supervised-learning & RL losses

---

## Implementation Details

* **NFSP** – replicates DeepMind's algorithm combining supervised learning of the average policy with DQN-based best-response training.
* **Ray actors** – every environment instance lives in `NFSPActor`, each holding local replay & reservoir buffers.  Gradients are shipped to the `Learner` for aggregation.
* **Evaluator** – runs asynchronously to avoid blocking training; computes metrics over 1 k games by default.
* **Dominoes environment** – `draw_dominoes` from OpenSpiel (two-player).  Swap in any other turn-based game by changing the `game` variable in `runner.py`.

## Results of Ahmed Attia Internship can be found on the following wandb project:
https://wandb.ai/ahmed-attia-mbzuai/nfsp-training?nw=nwuserahmedattia