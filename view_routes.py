"""
VRP-RL Route Viewer
===================
Runs inference on a trained model and saves:
  - routes.txt  : human-readable routes per vehicle (driver instructions)
  - routes.png  : visual map of the routes

Outputs are saved into the log folder of the model being evaluated.

WHAT ARE "PROBLEMS"?
  Each problem = one independent delivery scenario (one row in the test file).
  We test on many problems to measure average model performance.

USAGE:
  # Auto-finds latest trained model, uses default test data:
  python view_routes.py --is_train=False --task=vrp10

  # Use a specific trained model:
  python view_routes.py --is_train=False --task=vrp10 ^
      --load_path logs/vrp10-2026-03-12_17-53-52/model/model.ckpt-10000

  # Use your own custom CSV file:
  python view_routes.py --is_train=False --task=vrp10 ^
      --load_path logs/vrp10-.../model/model.ckpt-10000 ^
      --csv_path custom_testing/my_deliveries.csv

  # Control how many problems appear in routes.png (default=4):
  python view_routes.py --is_train=False --task=vrp10 --n_show 10

READ model_info.txt IN YOUR LOG FOLDER TO UNDERSTAND THE CSV FORMAT:
  logs/vrp10-.../model_info.txt
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import glob
import sys
import argparse

from configs import ParseParams
from model.attention_agent import RLAgent
from main import load_task_specific_components


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def find_latest_model():
    """Find the most recently trained model checkpoint."""
    log_dirs = sorted(glob.glob("logs/vrp*") + glob.glob("logs/tsp*"),
                      key=os.path.getmtime, reverse=True)
    for log_dir in log_dirs:
        model_dir = os.path.join(log_dir, "model")
        ckpt = tf.train.latest_checkpoint(model_dir)
        if ckpt:
            return ckpt, log_dir
    return None, None


def load_custom_csv(csv_path, n_nodes, task_name):
    """
    Load a custom CSV file and convert it into the model's expected format.
    Returns numpy array of shape [n_problems, n_nodes, input_dim].
    Raises ValueError with a clear message if the format is wrong.
    """
    import csv

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    input_dim = 3 if task_name == 'vrp' else 2
    expected_cols_per_node = input_dim
    expected_data_cols = n_nodes * expected_cols_per_node

    problems = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # skip header row

        # Count expected columns (excluding problem_id column)
        data_header_count = len(headers) - 1  # subtract problem_id col

        if data_header_count != expected_data_cols:
            raise ValueError(
                f"\n\nCSV FORMAT ERROR:\n"
                f"  Expected {expected_data_cols} data columns "
                f"({n_nodes} nodes × {expected_cols_per_node} values each)\n"
                f"  But got {data_header_count} data columns.\n\n"
                f"  Read the model_info.txt in your log folder for the exact format."
            )

        for row_num, row in enumerate(reader, start=2):
            if len(row) == 0:
                continue
            try:
                values = [float(v) for v in row[1:]]  # skip problem_id
            except ValueError as e:
                raise ValueError(f"Row {row_num}: Non-numeric value found: {e}")

            if len(values) != expected_data_cols:
                raise ValueError(
                    f"Row {row_num}: expected {expected_data_cols} values, got {len(values)}"
                )

            # Reshape to [n_nodes, input_dim]
            node_data = np.array(values).reshape(n_nodes, input_dim)

            # Validate coordinates are in [0, 1]
            coords = node_data[:, :2]
            if np.any(coords < 0) or np.any(coords > 1):
                raise ValueError(
                    f"Row {row_num}: All x,y coordinates must be between 0.0 and 1.0.\n"
                    f"  Found values outside range: min={coords.min():.3f}, max={coords.max():.3f}\n"
                    f"  Normalize your coordinates to [0,1] range first."
                )

            if task_name == 'vrp':
                # Validate depot demand = 0
                depot_demand = node_data[-1, 2]
                if depot_demand != 0:
                    raise ValueError(
                        f"Row {row_num}: The LAST node must be the depot with demand=0.\n"
                        f"  Got depot demand={depot_demand}"
                    )
                # Validate customer demands are positive integers
                cust_demands = node_data[:-1, 2]
                if np.any(cust_demands <= 0):
                    raise ValueError(
                        f"Row {row_num}: All customer demands must be >= 1.\n"
                        f"  Found demand <= 0."
                    )

            problems.append(node_data)

    if len(problems) == 0:
        raise ValueError(f"CSV file has no data rows: {csv_path}")

    data = np.array(problems, dtype=np.float32)
    print(f"  Loaded {len(problems)} problems from: {csv_path}")
    return data


def build_route_text(prob_idx, test_data, idx_sequence, R, capacity, source_label):
    """Build human-readable route text for one problem."""
    n_nodes = test_data.shape[1]
    n_cust = n_nodes - 1
    lines = []

    lines.append(f"\n{'═'*60}")
    lines.append(f"  PROBLEM {prob_idx + 1}  —  Total Distance: {R[prob_idx]:.3f}")
    lines.append(f"  Source: {source_label}, row {prob_idx + 1}")
    lines.append(f"{'═'*60}")
    lines.append(f"  Depot at: ({test_data[prob_idx, n_cust, 0]:.3f}, {test_data[prob_idx, n_cust, 1]:.3f})")

    lines.append(f"\n  Locations & Demands:")
    lines.append(f"  {'Node':<8} {'X':>8} {'Y':>8} {'Demand':>8}")
    lines.append(f"  {'─'*36}")
    for node in range(n_nodes):
        x = test_data[prob_idx, node, 0]
        y = test_data[prob_idx, node, 1]
        d = int(test_data[prob_idx, node, 2]) if test_data.shape[2] == 3 else 0
        label = "DEPOT" if node == n_cust else f"C{node+1}"
        suffix = "  <- vehicles start/end here" if node == n_cust else ""
        lines.append(f"  {label:<8} {x:>8.3f} {y:>8.3f} {d:>8}{suffix}")

    lines.append(f"\n  Routes for Drivers  (vehicle capacity: {capacity}):")
    route_num = 1
    current_load = 0
    route_nodes = []

    for step in range(idx_sequence.shape[1]):
        node_idx = int(idx_sequence[prob_idx, step])
        if node_idx == n_cust:
            if route_nodes:
                node_names = [f"C{n+1}" for n in route_nodes]
                demands = [int(test_data[prob_idx, n, 2]) for n in route_nodes] if test_data.shape[2] == 3 else []
                demands_str = " + ".join(str(d) for d in demands) if demands else "N/A"
                load_str = f"{demands_str} = {current_load}/{capacity}" if demands else ""
                lines.append(f"")
                lines.append(f"  Vehicle {route_num}:  Depot --> {' --> '.join(node_names)} --> Depot")
                if load_str:
                    lines.append(f"               Load: {load_str}")
                route_num += 1
                route_nodes = []
                current_load = 0
        else:
            route_nodes.append(node_idx)
            if test_data.shape[2] == 3:
                current_load += int(test_data[prob_idx, node_idx, 2])

    if route_nodes:
        node_names = [f"C{n+1}" for n in route_nodes]
        demands = [int(test_data[prob_idx, n, 2]) for n in route_nodes] if test_data.shape[2] == 3 else []
        demands_str = " + ".join(str(d) for d in demands) if demands else "N/A"
        load_str = f"{demands_str} = {current_load}/{capacity}" if demands else ""
        lines.append(f"")
        lines.append(f"  Vehicle {route_num}:  Depot --> {' --> '.join(node_names)} --> Depot")
        if load_str:
            lines.append(f"               Load: {load_str}")
        route_num += 1

    lines.append(f"\n  Total vehicles used: {route_num - 1}")
    return "\n".join(lines)


def save_route_map(test_data, idx_sequence, R, n_show, save_path):
    """Generate and save route visualization PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib not installed: pip install matplotlib")
        return

    n_nodes = test_data.shape[1]
    n_cust = n_nodes - 1
    actual_show = min(n_show, len(R))
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']

    cols = min(actual_show, 4)
    rows = (actual_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows + 0.5))
    axes_flat = np.array(axes).flatten() if actual_show > 1 else [axes]

    for prob_idx in range(actual_show):
        ax = axes_flat[prob_idx]
        dx = test_data[prob_idx, n_cust, 0]
        dy = test_data[prob_idx, n_cust, 1]

        # Draw routes
        route_num = 0
        prev_x, prev_y = dx, dy
        for step in range(idx_sequence.shape[1]):
            node_idx = int(idx_sequence[prob_idx, step])
            nx = test_data[prob_idx, node_idx, 0]
            ny = test_data[prob_idx, node_idx, 1]
            color = colors[route_num % len(colors)]
            if node_idx == n_cust:
                ax.plot([prev_x, nx], [prev_y, ny], color=color, linewidth=1.5, alpha=0.6)
                route_num += 1
                prev_x, prev_y = dx, dy
            else:
                ax.annotate('', xy=(nx, ny), xytext=(prev_x, prev_y),
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.8))
                prev_x, prev_y = nx, ny

        # Draw customers
        for node in range(n_cust):
            x = test_data[prob_idx, node, 0]
            y = test_data[prob_idx, node, 1]
            d = int(test_data[prob_idx, node, 2]) if test_data.shape[2] == 3 else ""
            label = f'C{node+1}\nd={d}' if d != "" else f'C{node+1}'
            ax.scatter(x, y, c='dodgerblue', s=120, zorder=5, edgecolors='black', linewidth=0.5)
            ax.annotate(label, (x, y), textcoords="offset points",
                       xytext=(0, 8), ha='center', fontsize=6.5, fontweight='bold')

        # Draw depot
        ax.scatter(dx, dy, c='red', s=220, marker='s', zorder=6, edgecolors='black', linewidth=1)
        ax.annotate('DEPOT', (dx, dy), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontsize=8, fontweight='bold', color='darkred')

        ax.set_title(f'Problem {prob_idx+1}  (row {prob_idx+1})\nDist: {R[prob_idx]:.3f}',
                    fontsize=9, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    for i in range(actual_show, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.suptitle(f'VRP-RL Generated Routes  (showing {actual_show} of {len(R)} problems)',
                fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def run_inference_and_get_routes():
    # ── Parse extra args before ParseParams consumes sys.argv ──
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--n_show', type=int, default=4)
    pre.add_argument('--csv_path', type=str, default='')
    extra, _ = pre.parse_known_args()
    n_show = extra.n_show
    csv_path = extra.csv_path

    args, prt = ParseParams()

    # ── Find model ──
    has_load = bool(args.get('load_path'))
    if not has_load:
        ckpt, model_log_dir = find_latest_model()
        if ckpt is None:
            print("No trained model found. Run: python main.py --task=vrp10 --n_train=10000")
            return
        args['load_path'] = os.path.dirname(ckpt)
        model_log_dir = os.path.dirname(os.path.normpath(args['load_path']))
        print(f"Auto-selected model: {ckpt}")
    else:
        model_log_dir = os.path.dirname(os.path.normpath(args['load_path']))

    output_dir = model_log_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Log what model and data we're using ──
    print(f"\n{'='*60}")
    print(f"  Model:      {args['load_path']}")
    print(f"  Task:       {args['task']}  ({args['n_cust']} customers, capacity={args.get('capacity','?')})")
    if csv_path:
        print(f"  Test data:  CUSTOM CSV → {csv_path}")
    else:
        print(f"  Test data:  default → data/vrp-size-*-len-*-test.txt")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*60}\n")

    # ── Load components ──
    DataGenerator, Env, reward_func, AttentionActor, AttentionCritic = \
        load_task_specific_components(args['task_name'])

    dataGen = DataGenerator(args)
    dataGen.reset()
    env = Env(args)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    agent = RLAgent(args, prt, env, dataGen, reward_func,
                    AttentionActor, AttentionCritic, is_train=False)
    agent.Initialize(sess)

    # ── Load test data (custom CSV or default) ──
    if csv_path:
        try:
            test_data = load_custom_csv(csv_path, args['n_nodes'], args['task_name'])
        except (ValueError, FileNotFoundError) as e:
            print(f"\n[ERROR] {e}")
            info_path = os.path.join(output_dir, "model_info.txt")
            if os.path.exists(info_path):
                print(f"\nRead {info_path} for the correct CSV format.")
            sess.close()
            return
        source_label = os.path.basename(csv_path)
    else:
        test_data = dataGen.get_test_all()
        source_label = f"data/vrp-size-{test_data.shape[0]}-len-{test_data.shape[1]}-test.txt"

    n_problems = test_data.shape[0]
    n_nodes = test_data.shape[1]
    n_cust = n_nodes - 1
    capacity = args.get('capacity', 999)

    print(f"Running inference on {n_problems} problems...")

    R, v, logprobs, actions, idxs, batch, _ = sess.run(
        agent.val_summary_greedy,
        feed_dict={env.input_data: test_data, agent.decodeStep.dropout: 0.0}
    )
    idx_sequence = np.concatenate(idxs, axis=1)

    # ── Save routes.txt ──
    routes_txt_path = os.path.join(output_dir, "routes.txt")
    lines = []
    lines.append("VRP-RL ROUTE OUTPUT")
    lines.append(f"{'='*60}")
    lines.append(f"Model dir:    {args['load_path']}")
    lines.append(f"Task:         {args['task']}  |  Customers: {n_cust}  |  Capacity: {capacity}")
    lines.append(f"Test source:  {source_label}")
    lines.append(f"Problems:     {n_problems}")
    lines.append(f"{'='*60}")
    lines.append(f"\nSUMMARY: Avg distance = {np.mean(R):.3f}  |  Std = {np.std(R):.3f}")
    lines.append(f"         Best = {np.min(R):.3f}  |  Worst = {np.max(R):.3f}")
    lines.append(f"\nNOTE: 'Problem N' = row N in the test source file above.")
    lines.append(f"      Each problem is an independent delivery scenario.")

    for prob_idx in range(n_problems):
        lines.append(build_route_text(prob_idx, test_data, idx_sequence, R, capacity, source_label))

    with open(routes_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"[✓] Routes text saved: {routes_txt_path}  ({n_problems} problems)")

    # ── Save routes.png ──
    png_path = os.path.join(output_dir, "routes.png")
    save_route_map(test_data, idx_sequence, R, n_show, png_path)
    print(f"[✓] Routes map  saved: {png_path}  (showing {min(n_show, n_problems)} problems)")
    print(f"    Use --n_show N to change how many appear in the image")

    # ── Quick terminal preview ──
    print("\n" + "="*60)
    print("  PREVIEW (first 2 problems):")
    print("="*60)
    for i in range(min(2, n_problems)):
        print(build_route_text(i, test_data, idx_sequence, R, capacity, source_label))

    sess.close()


if __name__ == '__main__':
    run_inference_and_get_routes()
