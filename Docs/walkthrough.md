# VRP-RL Training Results Analysis (VRP20)

This document provides a comparative analysis of the training runs for the Vehicle Routing Problem (20 customers) using four different Reinforcement Learning algorithms.

## Executive Summary

| Algorithm | Final Reward | Greedy Eval | Beam Search Eval | Training Time | Stability |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **REINFORCE** | **8.02** | **7.75** | **7.18** | **~4.9 hrs** | Highly Stable |
| **A2C** | 8.14 | 8.07 | 7.47 | ~5.1 hrs | Stable (Exploration+) |
| **Greedy Baseline** | 8.16 | 8.07 | 7.42 | ~19.8 hrs | Stable but Slow |
| **PPO** | 18.82 | 19.22 | 18.81 | ~4.3 hrs | **Failed (Divergent)** |

## Detailed Findings

### 1. REINFORCE (Best Performer)
REINFORCE with a learned critic baseline proved to be the most effective and efficient. It achieved the shortest routes (7.18 via Beam Search) in under 5 hours. The learning curve was smooth, and the critic loss minimized effectively (198.6 -> 0.45).

### 2. A2C (Advantage Actor-Critic)
A2C performed similarly to REINFORCE, slightly behind in final route quality. The entropy bonus in A2C encourages exploration, which might be beneficial for larger customer sizes, but for VRP20, REINFORCE was sufficient.

### 3. Greedy Baseline (Computational Trade-off)
The "Greedy Rollout Baseline" (Kool et al., 2019) achieved results nearly identical to REINFORCE but took **4x longer to train** (~20 hours vs 5 hours).
- **Why?** Every training step requires a full greedy decoding pass to compute the baseline reward.
- **Observation:** As noted in the logs, the `critic loss` is consistently `0.0` because there is no learned critic network; the baseline is the actual performance of the current best model.

### 4. PPO (Investigation needed)
PPO failed to converge, with route lengths actually *increasing* over time (14.1 -> 18.8). 
- **Symptoms:** Extremely small actor loss (-0.007) compared to others (-1.0 to -5.0). 
- **Probable Cause:** The policy might be getting stuck in local optima or "frozen" due to PPO's clipping mechanism, or the hyperparameters (clipping epsilon, epochs) are not tuned for VRP's sparse reward structure.

## Visualizations
The following files in each log directory provide further visual evidence:
- [training_plots.png](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/logs/vrp20-2026-03-14_11-45-01/training_plots.png): Stability and loss curves.
- [routes.png](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/logs/vrp20-2026-03-14_11-45-01/routes.png): Actual route visualizations for 25 test examples.

## Recommendations
- **Go-to Model:** Use **REINFORCE** for production/standard runs as it offers the best balance of speed and quality.
- **Debugging PPO:** If PPO is required, investigate the advantage scaling and clipping epsilon.
- **Large VRPs:** For larger tasks (VRP50+), the A2C entropy bonus might become more valuable to avoid local optima.
