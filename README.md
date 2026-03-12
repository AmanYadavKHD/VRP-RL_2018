# Reinforcement Learning for Solving the Vehicle Routing Problem

A TensorFlow implementation of the paper **[Reinforcement Learning for Solving the Vehicle Routing Problem](https://arxiv.org/abs/1802.04240v2)** (Nazari et al., 2018), updated for **TensorFlow 2.x** and enhanced with route visualization and custom data testing tools.

This code trains a neural network using **policy gradient methods** to solve **VRP** (Vehicle Routing Problem) and **TSP** (Travelling Salesman Problem) instances. Once trained, the model generates near-optimal delivery routes in milliseconds — no hand-crafted heuristics required.

---

## Table of Contents

- [What This Code Does](#what-this-code-does)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [RL Algorithm Selection](#rl-algorithm-selection)
- [Inference & Route Generation](#inference--route-generation)
- [Custom Data Testing](#custom-data-testing)
- [Analyzing Training Results](#analyzing-training-results)
- [Understanding the Output](#understanding-the-output)
- [Configuration Reference](#configuration-reference)
- [How the Model Works](#how-the-model-works)
- [Practical Use Cases](#practical-use-cases)
- [Limitations](#limitations)
- [Modernization Notes](#modernization-notes)
- [Acknowledgements](#acknowledgements)

---

## What This Code Does

Given a set of **customers** (with locations and delivery demands) and a **depot** (warehouse), the model finds **vehicle routes** that:

1. Visit every customer exactly once
2. Respect vehicle **capacity** constraints
3. Minimize **total travel distance**

**Input:** Customer coordinates + demands + depot location  
**Output:** Optimized routes per vehicle (e.g., `Depot → C3 → C7 → C1 → Depot`)

### Supported Problem Types

| Task | Description | Nodes | Capacity |
|------|-------------|-------|----------|
| `vrp10` | VRP with 10 customers | 11 (10 + depot) | 20 |
| `vrp20` | VRP with 20 customers | 21 (20 + depot) | 30 |
| `vrp50` | VRP with 50 customers | 51 (50 + depot) | 40 |
| `vrp100` | VRP with 100 customers | 101 (100 + depot) | 50 |
| `tsp10` | TSP with 10 cities | 10 | N/A |
| `tsp20` | TSP with 20 cities | 20 | N/A |
| `tsp50` | TSP with 50 cities | 50 | N/A |
| `tsp100` | TSP with 100 cities | 100 | N/A |

---

## Project Structure

```
VRP-RL_2018/
│
├── main.py                  # Entry point: training + inference
├── configs.py               # All hyperparameters and CLI argument parsing
├── task_specific_params.py  # Task definitions (n_nodes, capacity, demand)
│
├── model/                   # Core RL agent
│   └── attention_agent.py   # RLAgent class (actor-critic with attention)
│
├── shared/                  # Shared neural network components
│   ├── attention.py         # Multi-head attention mechanism
│   ├── decode_step.py       # RNN decoder with pointer network
│   ├── embeddings.py        # Input embedding layers
│   └── misc_utils.py        # Logging, entropy, distance utilities
│
├── VRP/                     # VRP-specific logic
│   ├── vrp_utils.py         # VRP environment, data generator, reward
│   └── vrp_attention.py     # VRP actor/critic attention layers
│
├── TSP/                     # TSP-specific logic
│   └── tsp_utils.py         # TSP environment, data generator, reward
│
├── rl_algorithms/           # Selectable RL training algorithms
│   ├── __init__.py          # Algorithm registry + get_algorithm() factory
│   ├── base.py              # Abstract base class
│   ├── reinforce.py         # REINFORCE with learned critic baseline
│   ├── a2c.py               # Advantage Actor-Critic + entropy bonus
│   ├── ppo.py               # Proximal Policy Optimization (clipped)
│   └── greedy_baseline.py   # Greedy Rollout Baseline (no critic)
│
├── view_routes.py           # Route visualization + driver instructions
├── analyze_results.py       # Training analysis plots + metrics
│
├── custom_testing/          # Your own test data goes here
│   ├── README.md            # Step-by-step guide for custom CSVs
│   ├── example_vrp10.csv    # Ready-to-use VRP example
│   └── example_tsp10.csv    # Ready-to-use TSP example
│
├── COMMANDS.md              # Full terminal command reference
│
├── data/                    # Auto-generated test datasets
├── logs/                    # Training outputs (one folder per run)
│   └── vrp10-YYYY-MM-DD_HH-MM-SS/
│       ├── model/           # Saved model checkpoints
│       ├── results.txt      # Training log + config
│       ├── model_info.txt   # CSV format docs + commands for this model
│       ├── routes.txt       # Generated vehicle routes
│       └── routes.png       # Route visualization map
│
└── misc_utils.py            # Legacy utilities (unused, kept for reference)
```

---

## Setup & Installation

### Requirements

- **Python** 3.8+
- **OS:** Windows / Linux / macOS

### Install Dependencies

```bash
pip install tensorflow numpy tqdm matplotlib scipy tf-keras
```

### Verify Installation

```bash
python main.py --task=vrp10 --n_train=2 --test_size=5
```

If this runs without errors, you're ready.

> **Note:** The code runs on **CPU by default** (no GPU required). For GPU, set `--gpu=0`.

---

## Quick Start

```bash
# 1. Train a model on 10-customer VRP (takes ~15 min on CPU)
python main.py --task=vrp10 --n_train=10000

# 2. See the routes the model generates
python view_routes.py --is_train=False --task=vrp10

# 3. Open the output files
#    logs/vrp10-.../routes.txt   → driver instructions
#    logs/vrp10-.../routes.png   → visual route map
```

---

## Training

### Basic Training

```bash
python main.py --task=vrp10 --n_train=10000
```

### Training Options

```bash
# Longer training (better quality, ~1-3 hrs on CPU)
python main.py --task=vrp10 --n_train=50000

# Larger batch size (if you have more RAM)
python main.py --task=vrp10 --n_train=10000 --batch_size=64

# 20-customer VRP
python main.py --task=vrp20 --n_train=10000

# TSP (no capacity constraint)
python main.py --task=tsp10 --n_train=10000

# Quick smoke test (2 training steps)
python main.py --task=vrp10 --n_train=2 --test_size=5
```

### What Training Produces

After training, a log folder is created:

```
logs/vrp10-2026-03-12_17-53-52/
├── model/              # Saved model weights (checkpoints)
├── results.txt         # Training config + step-by-step logs
└── model_info.txt      # How to use this model (CSV format, commands)
```

> **Read `model_info.txt`** — it contains everything you need to know about the model: constraints, CSV format, and copy-paste terminal commands.

---

## RL Algorithm Selection

The codebase supports **4 RL algorithms**, all sharing the same encoder-decoder pointer network. Choose via `--rl_model`:

```bash
python main.py --task=vrp10 --rl_model=reinforce         # default
python main.py --task=vrp10 --rl_model=a2c               # + entropy bonus
python main.py --task=vrp10 --rl_model=ppo               # clipped updates
python main.py --task=vrp10 --rl_model=greedy_baseline    # no critic needed
```

### Algorithm Comparison

| Algorithm | Loss Formula | Baseline | Best For |
|-----------|-------------|----------|----------|
| **reinforce** | `(R - V) × log π` | Learned critic | General baseline |
| **a2c** | `(R - V) × log π - β·H(π)` | Learned critic + entropy | Avoiding local optima |
| **ppo** | `min(ratio·A, clip(ratio)·A)` | Learned critic + clip | Stable training |
| **greedy_baseline** | `(R_sample - R_greedy) × log π` | Greedy rollout | Often best for routing |

### Algorithm Details

- **REINFORCE** (Nazari et al., 2018): Original paper's method. Simple policy gradient with learned value baseline.
- **A2C** (Advantage Actor-Critic): Adds entropy regularization (`--entropy_coeff`) to encourage exploration and escape local optima.
- **PPO** (Proximal Policy Optimization): Clips the policy update ratio (`--ppo_clip=0.2`) to prevent destructive large updates. Most stable training.
- **Greedy Baseline** (Kool et al., 2019): Uses greedy rollout reward instead of a learned critic. No critic network trained — simpler and often better for routing.

### Algorithm-Specific Flags

| Flag | Default | Used By |
|------|---------|---------|
| `--rl_model` | `reinforce` | All |
| `--entropy_coeff` | `0.01` | A2C, PPO |
| `--ppo_clip` | `0.2` | PPO |
| `--ppo_epochs` | `4` | PPO |

---

## Inference & Route Generation

After training, generate routes from the model:

```bash
# Auto-selects latest model + default test data
python view_routes.py --is_train=False --task=vrp10

# Specific model
python view_routes.py --is_train=False --task=vrp10 \
    --load_path "logs/vrp10-.../model/model.ckpt-10000"

# Control number of problems in the PNG (default: 4)
python view_routes.py --is_train=False --task=vrp10 --n_show 10
```

### Output Files

Saved to the model's log folder:

| File | Contents |
|------|----------|
| `routes.txt` | Vehicle-by-vehicle routes for every test problem |
| `routes.png` | Visual map showing routes on a 2D plane |

### Sample `routes.txt` Output

```
PROBLEM 1  —  Total Distance: 6.170
  Depot at: (0.500, 0.500)

  Locations & Demands:
  C1      0.200    0.800        3
  C2      0.400    0.900        5
  ...
  DEPOT   0.500    0.500        0  <- vehicles start/end here

  Routes for Drivers  (vehicle capacity: 20):

  Vehicle 1:  Depot --> C2 --> C10 --> C1 --> C3 --> Depot
               Load: 5 + 4 + 3 + 2 = 14/20

  Vehicle 2:  Depot --> C4 --> C5 --> Depot
               Load: 4 + 6 = 10/20
  ...
  Total vehicles used: 4
```

### Sample Route Map

Each colored line = one vehicle's route.  
🟥 Red square = Depot | 🔵 Blue circles = Customers

---

## Custom Data Testing

### Workflow

1. **Train** your model: `python main.py --task=vrp10 --n_train=10000`
2. **Read** `logs/vrp10-.../model_info.txt` — it tells you the exact CSV format
3. **Create** your CSV in `custom_testing/` (or copy the example)
4. **Run** inference with your CSV:

```bash
python view_routes.py --is_train=False --task=vrp10 \
    --csv_path custom_testing/my_deliveries.csv
```

### CSV Format (VRP10)

```csv
problem_id,C1_x,C1_y,C1_demand,C2_x,...,C10_demand,depot_x,depot_y,depot_demand
1,0.20,0.80,3,0.40,0.90,5,...,0.15,0.65,4,0.50,0.50,0
2,0.10,0.90,4,0.30,0.75,3,...,0.05,0.55,5,0.45,0.45,0
```

**Rules:**
- `x, y` coordinates must be in **[0.0, 1.0]** (normalize real GPS coordinates)
- Customer demands: integers **1 to 9** (for vrp10)
- Depot demand: must be **0**
- Last node in each row: always the **depot**
- Each row = one independent delivery problem

### Normalizing Real Coordinates

```python
# If you have GPS coordinates (lat/lon), normalize them:
x_norm = (longitude - lon_min) / (lon_max - lon_min)
y_norm = (latitude  - lat_min) / (lat_max - lat_min)
```

### Example Templates

Pre-made CSVs are in `custom_testing/`:
- `example_vrp10.csv` — 3 VRP problems with 10 customers each
- `example_tsp10.csv` — 3 TSP problems with 10 cities each

---

## Analyzing Training Results

```bash
# Analyze latest training run
python analyze_results.py

# Analyze a specific run
python analyze_results.py --log_dir logs/vrp10-2026-03-12_17-53-52
```

**Output:** `training_plots.png` in the log folder, containing:
- Training reward over time
- Actor and critic loss curves
- Greedy vs. beam search comparison

---

## Understanding the Output

### What is a "Problem"?

Each **problem** = one independent delivery scenario with `N` customers at random locations with random demands. The test file contains many problems (default: 1000) to measure average model performance.

### What is "Distance"?

The total route distance across all vehicles, measured in normalized units (coordinates are in [0, 1] space). **Lower = better**.

### How the Model Auto-Selects

| What | How it's auto-selected |
|------|----------------------|
| **Model** | Most recently modified `logs/*/model/` folder |
| **Test data** | Generated once with fixed random seed in `data/` |
| **Output folder** | Saved to the evaluated model's log directory |

> **For reproducibility:** always specify `--load_path` explicitly.

---

## Configuration Reference

### Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | `vrp10` | Problem type (`vrp10`, `vrp20`, `tsp10`, etc.) |
| `--n_train` | `10000` | Number of training steps |
| `--batch_size` | `32` | Problems per training batch |
| `--test_size` | `1000` | Number of test problems |
| `--is_train` | `True` | `True` = train, `False` = inference |
| `--load_path` | *(empty)* | Path to model checkpoint |
| `--gpu` | `""` | GPU ID (empty = CPU) |
| `--n_show` | `4` | Problems shown in routes.png |
| `--csv_path` | `""` | Path to custom CSV test file |

For the full list, see `configs.py` and `task_specific_params.py`.

---

## How the Model Works

### Architecture

```
Input (customer locations + demands)
    ↓
Embedding Layer (Conv1D)
    ↓
Encoder (Multi-head Attention × 3 blocks)
    ↓
Decoder (LSTM + Pointer Network)
    ↓
Output (sequence of node visits = route)
```

### Training Method

- **Actor-Critic** policy gradient (REINFORCE with baseline)
- **Actor** learns to select the next customer to visit
- **Critic** estimates the expected route distance (baseline for variance reduction)
- **Reward** = negative total route distance (maximizing reward = minimizing distance)

### Inference Strategies

- **Greedy:** Always pick the highest-probability next node
- **Beam Search:** Explore top-K candidates at each step (better quality, slower)

---

## Practical Use Cases

### Who Uses This?

| Role | How They Use It |
|------|----------------|
| **Logistics planner** | Runs model daily before dispatching to get optimal routes |
| **Researcher** | Benchmarks RL vs. classical VRP solvers (OR-Tools, etc.) |
| **Student** | Learns how RL applies to combinatorial optimization |

### When to Run

- **Before** delivery, not during (it's an offline optimizer)
- Run **daily** or whenever delivery points change
- Train **once**, then run inference as many times as needed

### What This Does NOT Do

- Does **not** use real maps or road networks (works in abstract 2D space)
- Does **not** handle real-time traffic
- Does **not** consider time windows or driver schedules
- Does **not** have a user interface (command-line only)

For production use, this model would be one component in a larger system that includes geocoding, map APIs, and a dispatcher UI.

---

## Limitations

1. **Abstract coordinates** — operates in [0,1] × [0,1] unit square, not real geography
2. **Euclidean distance** — straight-line, not road distance
3. **Fixed problem size** — a model trained on vrp10 cannot solve vrp20
4. **No time windows** — doesn't consider delivery deadlines
5. **Unlimited vehicles** — doesn't limit fleet size
6. **CPU performance** — training on CPU is slow; GPU recommended for larger problems
7. **Small-scale only** — practical for ≤100 customers; larger instances need more training

---

## Modernization Notes

This codebase was originally written for **TensorFlow 1.x** (2018). The following updates were made for compatibility with **TensorFlow 2.x**:

| Change | Details |
|--------|---------|
| TF imports | `import tensorflow as tf` → `import tensorflow.compat.v1 as tf` |
| v2 behavior | Added `tf.disable_v2_behavior()` |
| `tf.contrib` | Replaced with native TF2 equivalents |
| `tf.layers` | Migrated to `tf.keras.layers` |
| Layer names | Added `.replace('/', '_')` to avoid Keras 3 `ValueError` |
| Legacy RNN | Installed `tf-keras`, set `TF_USE_LEGACY_KERAS=1` |
| CPU mode | Default `--gpu=""` for CPU-only execution |
| Batch size | Reduced from 128 to 32 for low-RAM systems |

### Environment Variable

```python
# Set BEFORE importing TensorFlow (already done in main.py and view_routes.py)
os.environ['TF_USE_LEGACY_KERAS'] = '1'
```

---

## Acknowledgements

- **Paper:** [Reinforcement Learning for Solving the Vehicle Routing Problem](https://arxiv.org/abs/1802.04240v2) — Nazari et al., NeurIPS 2018
- **Code structure inspired by:** [pemami4911/neural-combinatorial-rl-pytorch](https://github.com/pemami4911/neural-combinatorial-rl-pytorch)
- **TF2 modernization, route visualization, and custom testing tools** added as part of coursework extensions
