"""
RL Algorithms for VRP/TSP Pointer Networks.

All algorithms share the same encoder-decoder attention architecture.
They only differ in how the training loss is computed and gradients applied.

Usage:
    from rl_algorithms import get_algorithm
    algo = get_algorithm('ppo')  # returns PPO class
"""

from rl_algorithms.reinforce import REINFORCE
from rl_algorithms.a2c import A2C
from rl_algorithms.ppo import PPO
from rl_algorithms.greedy_baseline import GreedyBaseline

ALGORITHMS = {
    'reinforce': REINFORCE,
    'a2c': A2C,
    'ppo': PPO,
    'greedy_baseline': GreedyBaseline,
}

def get_algorithm(name):
    """Get an RL algorithm class by name."""
    if name not in ALGORITHMS:
        available = ', '.join(ALGORITHMS.keys())
        raise ValueError(f"Unknown RL algorithm: '{name}'. Available: {available}")
    return ALGORITHMS[name]

def list_algorithms():
    """Return dict of {name: description} for all algorithms."""
    return {name: cls.description() for name, cls in ALGORITHMS.items()}
