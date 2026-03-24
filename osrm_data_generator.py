"""
Unified OSRM Time Matrix Generator
----------------------------------
Generates datasets for TSP and VRP training.

Modes:
  --mode osrm     : Fetch real driving times from OSRM API (slow, needs internet)
  --mode fallback : Use Euclidean * scale as surrogate travel time (fast, offline)

Usage:
  python osrm_data_generator.py --task vrp10 --n 300 --mode fallback
  python osrm_data_generator.py --task tsp10 --n 300 --mode fallback
  python osrm_data_generator.py --task vrp10 --n 10  --mode osrm
"""

import numpy as np
import requests
import time as time_module
import os
import json
import argparse
import math

# OSRM public demo server (for small-scale real-world data)
LAT_MIN, LAT_MAX = 40.70, 40.80
LON_MIN, LON_MAX = -74.01, -73.91
OSRM_URL = "http://router.project-osrm.org/table/v1/driving/"


def fetch_osrm_time_matrix(coords):
    """Fetch real driving-time matrix from OSRM. coords = list of (lon, lat)."""
    coords_str = ";".join([f"{lon:.5f},{lat:.5f}" for lon, lat in coords])
    url = f"{OSRM_URL}{coords_str}?annotations=duration"
    try:
        t0 = time_module.time()
        response = requests.get(url, timeout=15)
        elapsed = time_module.time() - t0
        data = response.json()
        code = data.get("code", "??")
        if code == "Ok":
            mat = np.array(data["durations"])
            print(f"    OSRM OK in {elapsed:.1f}s | matrix[0][1]={mat[0][1]:.1f}s, matrix shape={mat.shape}")
            return mat
        else:
            print(f"    OSRM returned code='{code}' in {elapsed:.1f}s | message: {data.get('message','')}")
            return None
    except Exception as e:
        print(f"    OSRM FAILED: {e}")
        return None


def euclidean_time_matrix(coords, speed_factor=600.0):
    """Compute surrogate travel-time matrix from normalised [0,1] coordinates.
    speed_factor=600 means 1.0 unit distance = 600 seconds (10 min)."""
    n = len(coords)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                mat[i][j] = round(math.sqrt(dx*dx + dy*dy) * speed_factor, 2)
    return mat


def generate_dataset(task, n_problems, mode, seed=42):
    """Generate dataset for a given task."""
    rng = np.random.RandomState(seed)

    is_vrp = task.startswith('vrp')
    if is_vrp:
        n_cust = int(task[3:])
        n_nodes = n_cust + 1
        demand_max = 9
    else:
        n_nodes = int(task[3:])

    dataset = []
    print(f"Generating {n_problems} {task.upper()} instances (mode={mode}, n_nodes={n_nodes})...")

    for i in range(n_problems):
        if mode == 'osrm':
            # Real GPS coords → OSRM
            lats = rng.uniform(LAT_MIN, LAT_MAX, n_nodes)
            lons = rng.uniform(LON_MIN, LON_MAX, n_nodes)
            gps_coords = list(zip(lons.tolist(), lats.tolist()))
            print(f"  [{i+1}/{n_problems}] Fetching from OSRM ({n_nodes} nodes)...")
            matrix = fetch_osrm_time_matrix(gps_coords)
            if matrix is None:
                print(f"  [{i+1}/{n_problems}] OSRM failed, using Manhattan fallback")
                matrix = np.zeros((n_nodes, n_nodes))
                for r in range(n_nodes):
                    for c in range(n_nodes):
                        matrix[r][c] = (abs(lats[r]-lats[c]) + abs(lons[r]-lons[c])) * 100000
            # Normalise GPS coords to [0,1] for neural network input
            coords = [
                [round(float((lons[k]-LON_MIN)/(LON_MAX-LON_MIN)), 4),
                 round(float((lats[k]-LAT_MIN)/(LAT_MAX-LAT_MIN)), 4)]
                for k in range(n_nodes)
            ]
            time_module.sleep(0.5)  # rate-limit OSRM
        else:
            # Fast fallback: random [0,1] coords with Euclidean surrogate times
            coords = [[round(float(rng.uniform(0, 1)), 4),
                        round(float(rng.uniform(0, 1)), 4)]
                       for _ in range(n_nodes)]
            matrix = euclidean_time_matrix(coords)

        instance = {
            "coordinates": coords,
            "time_matrix": matrix.tolist()
        }

        if is_vrp:
            demands = rng.randint(1, demand_max + 1, n_nodes).tolist()
            demands[-1] = 0  # depot has no demand
            instance["demands"] = demands

        dataset.append(instance)

        if (i+1) % 50 == 0:
            print(f"  ... generated {i+1}/{n_problems}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate OSRM time-matrix datasets")
    parser.add_argument('--task', type=str, default='vrp10',
                        help='Task: vrp10, vrp20, vrp50, tsp9, tsp10, tsp20')
    parser.add_argument('--n', type=int, default=300,
                        help='Number of problem instances')
    parser.add_argument('--mode', type=str, default='fallback',
                        choices=['osrm', 'fallback'],
                        help='osrm=real API, fallback=Euclidean surrogate')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='data')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dataset = generate_dataset(args.task, args.n, args.mode, args.seed)

    out_path = os.path.join(args.out_dir, f'osrm_{args.task}.json')
    with open(out_path, 'w') as f:
        json.dump(dataset, f)
    print(f"[OK] Saved {len(dataset)} instances → {out_path}")


if __name__ == '__main__':
    main()
