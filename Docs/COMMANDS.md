# VRP-RL — Complete Terminal Command Reference

All commands are run from the project root folder:
```
cd "c:\COLLEGE\SEM X\WORK\VRP-RL_2018"
```

---

## How the Code Picks Model & Data (When You Don't Specify)

| What's auto-selected | How |
|---|---|
| **Model** | Most recently modified `logs/*/model/` folder with a checkpoint |
| **Test data** | `data/vrp-size-{test_size}-len-{n_nodes}-test.txt` (generated once with fixed random seed) |
| **Output folder** | Always inside the model's log folder (`logs/vrp10-.../`) |

> To be explicit and reproducible, always specify `--load_path`.

---

## Training Commands

```powershell
# Train vrp10 (10 customers) — recommended for laptop
python main.py --task=vrp10 --n_train=10000

# Train for longer (better quality, ~3-4hrs on CPU)
python main.py --task=vrp10 --n_train=50000

# Train with more memory if you have RAM
python main.py --task=vrp10 --n_train=10000 --batch_size=64

# Train vrp20 (20 customers — heavier)
python main.py --task=vrp20 --n_train=10000

# Train TSP (no capacity constraint)
python main.py --task=tsp10 --n_train=10000

# Train and test on smaller test set (faster)
python main.py --task=vrp10 --n_train=10000 --test_size=100

# Quick smoke test (just 2 steps, to check things work)
python main.py --task=vrp10 --n_train=2 --test_size=10
```

**After training, outputs go to:**
```
logs/vrp10-YYYY-MM-DD_HH-MM-SS/
├── results.txt       ← Training log + metrics
├── model_info.txt    ← CSV format docs + commands  ← READ THIS
├── model/            ← Saved model weights
```

---

## Inference / Route Generation Commands

```powershell
# Auto-select latest model, default test data:
python view_routes.py --is_train=False --task=vrp10

# Specific model (recommended):
python view_routes.py --is_train=False --task=vrp10 `
    --load_path "logs/vrp10-2026-03-12_17-53-52/model/model.ckpt-10000"

# Your own CSV data:
python view_routes.py --is_train=False --task=vrp10 `
    --load_path "logs/vrp10-2026-03-12_17-53-52/model/model.ckpt-10000" `
    --csv_path custom_testing/my_deliveries.csv

# Change number of problems shown in routes.png (default=4):
python view_routes.py --is_train=False --task=vrp10 --n_show 10

# Show all problems in PNG:
python view_routes.py --is_train=False --task=vrp10 --n_show 1000
```

**Outputs saved to the model's log folder:**
```
logs/vrp10-.../
├── routes.txt    ← All vehicle routes, problem by problem
└── routes.png    ← Visual map (--n_show controls how many)
```

---

## Analysis Commands

```powershell
# Analyze latest training run (auto-selected):
python analyze_results.py

# Analyze specific training run:
python analyze_results.py --log_dir logs/vrp10-2026-03-12_17-53-52

# Output: logs/vrp10-.../training_plots.png
```

---

## Key Flags Reference

| Flag | Default | What it does |
|---|---|---|
| `--task` | `vrp10` | Problem type: `vrp10`, `vrp20`, `vrp50`, `tsp10`, `tsp20`... |
| `--n_train` | `10000` | Number of training steps |
| `--batch_size` | `32` | Problems per training batch (reduce if low RAM) |
| `--test_size` | `1000` | Number of test problems in default test set |
| `--is_train` | `True` | `True`=train, `False`=inference only |
| `--load_path` | *(empty)* | Path to model checkpoint to load |
| `--gpu` | `""` | GPU number, empty=CPU |
| `--n_show` | `4` | Problems in routes.png (view_routes.py only) |
| `--csv_path` | `""` | Custom CSV file (view_routes.py only) |

---

## Understanding the Log Folder Name

```
logs/vrp10-2026-03-12_17-53-52/
      ^^^^  ^^^^^^^^^^^^^^^^^^
      task  timestamp of when training started
```

Multiple trainings of the same task create separate folders. The one with the latest timestamp is the most recent.

---

## File Locations Summary

| File | Purpose |
|---|---|
| `main.py` | Training entry point |
| `view_routes.py` | Inference + route output |
| `analyze_results.py` | Training charts + stats |
| `configs.py` | All hyperparameter defaults |
| `task_specific_params.py` | Task definitions (n_nodes, capacity, etc.) |
| `data/` | Auto-generated test data files |
| `logs/` | All training runs (one folder each) |
| `custom_testing/` | Your own CSV test files go here |
| `custom_testing/README.md` | How to build custom CSV |
| `logs/*/model_info.txt` | Per-model CSV format + commands |






#Testing multiple RL Models
1. REINFORCE (The Baseline)
bash
python main.py --task vrp20 --rl_model reinforce --n_train 30000 --batch_size 32 --test_size 25 --n_glimpses 1
2. A2C (Fast & Stable)
bash
python main.py --task vrp20 --rl_model a2c --n_train 30000 --batch_size 32 --test_size 25 --n_glimpses 1
3. PPO (Advanced learning)
bash
python main.py --task vrp20 --rl_model ppo --n_train 30000 --batch_size 32 --test_size 25 --n_glimpses 1
4. Greedy Baseline (The Routing Specialist)
bash
python main.py --task vrp20 --rl_model greedy_baseline --n_train 30000 --batch_size 32 --test_size 25 --n_glimpses 1