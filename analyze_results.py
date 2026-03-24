"""
VRP-RL Results Analyzer
=======================
Run this after training to visualize and understand your results.

Usage:
    python analyze_results.py                           # auto-finds latest log
    python analyze_results.py --log_dir logs/vrp10-...  # specific run
"""

import os
import re
import argparse
import glob

def find_latest_log():
    """Find the most recent log directory."""
    log_dirs = glob.glob("logs/vrp*") + glob.glob("logs/tsp*")
    if not log_dirs:
        print("No log directories found! Run training first.")
        return None
    latest = max(log_dirs, key=os.path.getmtime)
    return latest

def parse_results(log_dir):
    """Parse results.txt and extract training metrics."""
    results_file = os.path.join(log_dir, "results.txt")
    if not os.path.exists(results_file):
        print(f"No results.txt found in {log_dir}")
        return None

    steps = []
    train_rewards = []
    values = []
    actor_losses = []
    critic_losses = []
    greedy_avgs = []
    beam_avgs = []
    greedy_stds = []
    beam_stds = []
    config = {}

    with open(results_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()

        # Parse config lines (key: value)
        config_match = re.match(r'^(\w+): (.+)$', line)
        if config_match and 'Train Step' not in line and 'Average' not in line:
            config[config_match.group(1)] = config_match.group(2)

        # Parse training step lines
        train_match = re.search(
            r'Train Step: (\d+) -- Time: .+ -- Train reward: ([\d.e+-]+) -- Value: ([\d.e+-]+)',
            line
        )
        if train_match:
            steps.append(int(train_match.group(1)))
            train_rewards.append(float(train_match.group(2)))
            values.append(float(train_match.group(3)))

        # Parse loss lines
        loss_match = re.search(
            r'actor loss: ([\d.e+-]+) -- critic loss: ([\d.e+-]+)',
            line
        )
        if loss_match:
            actor_losses.append(float(loss_match.group(1)))
            critic_losses.append(float(loss_match.group(2)))

        # Parse greedy evaluation
        greedy_match = re.search(
            r'Average of greedy.*?: ([\d.e+-]+) -- std ([\d.e+-]+)',
            line
        )
        if greedy_match:
            greedy_avgs.append(float(greedy_match.group(1)))
            greedy_stds.append(float(greedy_match.group(2)))

        # Parse beam search evaluation
        beam_match = re.search(
            r'Average of beam_search.*?: ([\d.e+-]+) -- std ([\d.e+-]+)',
            line
        )
        if beam_match:
            beam_avgs.append(float(beam_match.group(1)))
            beam_stds.append(float(beam_match.group(2)))

    return {
        'steps': steps,
        'train_rewards': train_rewards,
        'values': values,
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'greedy_avgs': greedy_avgs,
        'greedy_stds': greedy_stds,
        'beam_avgs': beam_avgs,
        'beam_stds': beam_stds,
        'config': config
    }

def print_summary(data, log_dir):
    """Print a human-readable summary of the training run."""
    print("=" * 65)
    print(f"  VRP-RL RESULTS ANALYSIS")
    print(f"  Log directory: {log_dir}")
    print("=" * 65)

    config = data['config']
    print(f"\n--- Configuration ---")
    print(f"  Task:          {config.get('task', '?')}")
    print(f"  Customers:     {config.get('n_cust', '?')}")
    print(f"  Capacity:      {config.get('capacity', '?')}")
    print(f"  Training steps:{config.get('n_train', '?')}")
    print(f"  Batch size:    {config.get('batch_size', '?')}")
    print(f"  Hidden dim:    {config.get('hidden_dim', '?')}")

    if data['train_rewards']:
        print(f"\n--- Training Progress ---")
        print(f"  Steps logged:     {len(data['steps'])}")
        print(f"  First reward:     {data['train_rewards'][0]:.3f}")
        print(f"  Last reward:      {data['train_rewards'][-1]:.3f}")
        improvement = data['train_rewards'][0] - data['train_rewards'][-1]
        pct = (improvement / data['train_rewards'][0]) * 100 if data['train_rewards'][0] != 0 else 0
        if improvement > 0:
            print(f"  Improvement:      {improvement:.1f}s ({pct:.1f}% faster routes)")
        else:
            print(f"  Change:           {improvement:.3f} (model hasn't improved yet)")

    if data['greedy_avgs']:
        print(f"\n--- Evaluation (lower = faster routes) ---")
        print(f"  Greedy  (latest): {data['greedy_avgs'][-1]:.3f} ± {data['greedy_stds'][-1]:.3f}")
        if data['beam_avgs']:
            print(f"  Beam Search:      {data['beam_avgs'][-1]:.3f} ± {data['beam_stds'][-1]:.3f}")
            gap = data['greedy_avgs'][-1] - data['beam_avgs'][-1]
            print(f"  Beam improvement: {gap:.3f} better than greedy")

    if data['critic_losses']:
        print(f"\n--- Loss ---")
        print(f"  First critic loss: {data['critic_losses'][0]:.3f}")
        print(f"  Last critic loss:  {data['critic_losses'][-1]:.3f}")

    print(f"\n{'=' * 65}")

def plot_results(data, log_dir):
    """Generate training plots and save them."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[!] matplotlib not installed. Install it for plots:")
        print("    pip install matplotlib")
        return

    if len(data['steps']) < 2:
        print("\n[!] Not enough data points to plot (need at least 2 training steps logged).")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"VRP-RL Training Analysis — {os.path.basename(log_dir)}", fontsize=14, fontweight='bold')

    # 1. Training Reward (travel time) over time
    ax = axes[0, 0]
    ax.plot(data['steps'], data['train_rewards'], 'b-', linewidth=1.5, label='Train reward')
    if data['values']:
        ax.plot(data['steps'], data['values'], 'r--', linewidth=1, alpha=0.7, label='Critic value')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Travel Time (seconds)')
    ax.set_title('Training Reward — Travel Time (↓ lower = better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Actor & Critic Loss
    ax = axes[0, 1]
    if data['actor_losses']:
        ax.plot(data['steps'], data['actor_losses'], 'g-', linewidth=1, label='Actor loss')
    if data['critic_losses']:
        ax2 = ax.twinx()
        ax2.plot(data['steps'], data['critic_losses'], 'orange', linewidth=1, label='Critic loss')
        ax2.set_ylabel('Critic Loss', color='orange')
        ax2.legend(loc='upper right')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Actor Loss', color='g')
    ax.set_title('Losses Over Training')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # 3. Greedy vs Beam Search evaluation
    ax = axes[1, 0]
    if data['greedy_avgs']:
        eval_steps = list(range(len(data['greedy_avgs'])))
        ax.plot(eval_steps, data['greedy_avgs'], 'b-o', markersize=4, label='Greedy')
        ax.fill_between(eval_steps,
                        [a - s for a, s in zip(data['greedy_avgs'], data['greedy_stds'])],
                        [a + s for a, s in zip(data['greedy_avgs'], data['greedy_stds'])],
                        alpha=0.2, color='blue')
    if data['beam_avgs']:
        ax.plot(eval_steps[:len(data['beam_avgs'])], data['beam_avgs'], 'r-o', markersize=4, label='Beam Search')
        ax.fill_between(eval_steps[:len(data['beam_avgs'])],
                        [a - s for a, s in zip(data['beam_avgs'], data['beam_stds'])],
                        [a + s for a, s in zip(data['beam_avgs'], data['beam_stds'])],
                        alpha=0.2, color='red')
    ax.set_xlabel('Evaluation Number')
    ax.set_ylabel('Avg Travel Time (seconds)')
    ax.set_title('Evaluation: Greedy vs Beam Search (↓ lower = better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Reward improvement histogram
    ax = axes[1, 1]
    if len(data['train_rewards']) > 10:
        n = len(data['train_rewards'])
        chunk = max(n // 5, 1)
        chunks = [data['train_rewards'][i:i + chunk] for i in range(0, n, chunk)]
        chunk_means = [sum(c) / len(c) for c in chunks if c]
        labels = [f"Steps\n{i * chunk * (data['steps'][1] - data['steps'][0]) if len(data['steps']) > 1 else 0}-{(i + 1) * chunk * (data['steps'][1] - data['steps'][0]) if len(data['steps']) > 1 else 0}" for i in range(len(chunk_means))]
        colors = plt.cm.RdYlGn_r([i / len(chunk_means) for i in range(len(chunk_means))])
        ax.bar(range(len(chunk_means)), chunk_means, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Avg Travel Time (seconds)')
        ax.set_title('Travel Time by Training Phase (↓ lower = better)')
        ax.set_xticks(range(len(chunk_means)))
        ax.set_xticklabels([f"Phase {i+1}" for i in range(len(chunk_means))])
    else:
        ax.text(0.5, 0.5, 'Not enough data\nfor histogram', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Reward by Training Phase')

    plt.tight_layout()
    save_path = os.path.join(log_dir, "training_plots.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[✓] Plots saved to: {save_path}")
    print(f"    Open this file to see your training charts!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='', help='Path to log directory')
    args = parser.parse_args()

    log_dir = args.log_dir if args.log_dir else find_latest_log()
    if log_dir is None:
        exit(1)

    print(f"Analyzing: {log_dir}\n")
    data = parse_results(log_dir)
    if data is None:
        exit(1)

    print_summary(data, log_dir)
    plot_results(data, log_dir)
