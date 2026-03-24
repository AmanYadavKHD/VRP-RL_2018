# Multi-RL Model Integration Plan

## Goal

Add 4 selectable RL training algorithms to the codebase, chosen via `--rl_model` flag:

```bash
python main.py --task=vrp10 --rl_model=reinforce    # current default
python main.py --task=vrp10 --rl_model=a2c          # Advantage Actor-Critic
python main.py --task=vrp10 --rl_model=ppo          # Proximal Policy Optimization
python main.py --task=vrp10 --rl_model=greedy_baseline  # Greedy Rollout Baseline
```

## Key Design Decision

All 4 algorithms share the **same encoder-decoder pointer network** — they only differ in how the **loss is computed** and **gradients are applied**. The architecture stays identical.

| Algorithm | Actor Loss | Baseline | Extra Params |
|-----------|-----------|----------|-------------|
| **REINFORCE** (current) | [(R - V) × log π](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/TSP/tsp_utils.py#111-167) | Learned critic network | None |
| **A2C** | `advantage × log π + entropy_bonus` | Learned critic, n-step | `entropy_coeff` |
| **PPO** | `min(ratio × A, clip(ratio) × A)` | Learned critic | `ppo_clip=0.2`, `ppo_epochs=4` |
| **Greedy Baseline** | [(R - R_greedy) × log π](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/TSP/tsp_utils.py#111-167) | Greedy rollout (no critic) | `baseline_update=100` |

---

## Proposed Changes

### 1. New folder: `rl_algorithms/`

#### [NEW] `rl_algorithms/__init__.py`
Exports `get_algorithm(name)` → returns the right class.

#### [NEW] `rl_algorithms/base.py`
Abstract base class defining the interface:
```python
class RLAlgorithm:
    def build_train_step(agent, train_summary) → train_op
    def get_description() → str
```

#### [NEW] `rl_algorithms/reinforce.py`
Extracts existing REINFORCE logic from [attention_agent.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py) lines 254-302.

#### [NEW] `rl_algorithms/a2c.py`
Advantage Actor-Critic: adds entropy regularization to the actor loss.

#### [NEW] `rl_algorithms/ppo.py`
PPO: stores old log-probs, computes ratio, clips the objective.

#### [NEW] `rl_algorithms/greedy_baseline.py`
Uses greedy rollout as baseline instead of learned critic.

---

### 2. Modify existing files

#### [MODIFY] [configs.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/configs.py)
Add `--rl_model` argument (choices: `reinforce`, `a2c`, `ppo`, `greedy_baseline`, default: `reinforce`).
Add algorithm-specific params (`--ppo_clip`, `--ppo_epochs`, `--entropy_coeff`, `--baseline_update_interval`).

#### [MODIFY] [model/attention_agent.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py)
- [build_train_step()](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py#254-303) delegates to the selected `RLAlgorithm` class
- Constructor takes `rl_algorithm` parameter
- Critic is still built in [build_model()](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py#72-253) but only when the algorithm needs it (not for `greedy_baseline`)

#### [MODIFY] [main.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/main.py)
- Import and instantiate the selected RL algorithm
- Pass it to [RLAgent](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py#8-401)
- Log which algorithm is being used in [model_info.txt](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/logs/vrp10-2026-03-12_22-18-37/model_info.txt)

#### [MODIFY] [README.md](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/README.md)
- Add section on RL algorithm selection

---

## Algorithm Details

### 1. REINFORCE with Learned Baseline (existing)
```
actor_loss  = mean( (R - V_critic) × Σ log π(a|s) )
critic_loss = MSE(R, V_critic)
```
- Already implemented. Will be extracted into `rl_algorithms/reinforce.py`.

### 2. A2C (Advantage Actor-Critic)
```
advantage   = R - V_critic
actor_loss  = mean( advantage × Σ log π(a|s) ) - entropy_coeff × entropy(π)
critic_loss = MSE(R, V_critic)
```
- Adds entropy bonus to encourage exploration
- Uses the same critic as REINFORCE, so baseline section in [build_model](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py#72-253) is reused

### 3. PPO (Proximal Policy Optimization)
```
ratio       = exp( log π_new - log π_old )
advantage   = R - V_critic
surrogate1  = ratio × advantage
surrogate2  = clip(ratio, 1-ε, 1+ε) × advantage
actor_loss  = -mean( min(surrogate1, surrogate2) ) - entropy_coeff × entropy
critic_loss = MSE(R, V_critic)
```
- Requires storing old log-probs from the previous iteration
- Multiple gradient update epochs per batch (`ppo_epochs`)
- Clips the policy ratio to prevent destructive updates

### 4. Greedy Rollout Baseline (Kool et al., 2019)
```
R_greedy    = reward from greedy decoding of the SAME batch
actor_loss  = mean( (R - R_greedy) × Σ log π(a|s) )
```
- No learned critic at all → simpler, sometimes better
- Periodically updates the baseline model (every `baseline_update_interval` steps) by copying current model weights
- Needs a second forward pass (greedy) on the same batch for the baseline

---

## Verification Plan

### Automated Tests
```bash
# Test each algorithm starts and runs 2 steps without errors
python main.py --task=vrp10 --rl_model=reinforce --n_train=2 --test_size=5
python main.py --task=vrp10 --rl_model=a2c --n_train=2 --test_size=5
python main.py --task=vrp10 --rl_model=ppo --n_train=2 --test_size=5
python main.py --task=vrp10 --rl_model=greedy_baseline --n_train=2 --test_size=5
```

### Expected Behavior
- All 4 should complete with exit code 0
- [model_info.txt](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/logs/vrp10-2026-03-12_22-18-37/model_info.txt) should mention which algorithm was used
- [view_routes.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/view_routes.py) should work with all 4 models (inference is the same)
