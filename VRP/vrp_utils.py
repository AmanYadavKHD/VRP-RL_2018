import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import collections
import json
import math
import os

# ──────────────────────────────────────────────────────────────
# Utility: compute Euclidean-based surrogate time matrix
# ──────────────────────────────────────────────────────────────
def compute_time_matrix(coords, speed_factor=600.0):
    """Given [n_nodes x 2] coords in [0,1], return [n_nodes x n_nodes] time matrix (seconds).
    speed_factor=600 means 1.0 unit = 600 seconds (~10 min)."""
    n = len(coords)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                mat[i][j] = math.sqrt(dx*dx + dy*dy) * speed_factor
    return mat


class DataGenerator(object):
    def __init__(self, args):
        '''
        Phase 1 DataGenerator for VRP with Time Matrices.
        - Training: generates random coords each step (infinite diversity) 
          + computes time matrix on-the-fly.
        - Testing: loads fixed OSRM JSON dataset from data/ folder.
        '''
        self.args = args
        self.rnd = np.random.RandomState(seed=args.get('random_seed', 1234))
        self.n_cust = args['n_cust']
        self.n_nodes = args['n_nodes']
        self.demand_max = args.get('demand_max', 9)
        
        # Load OSRM test dataset
        dataset_path = os.path.join("data", f"osrm_vrp{self.n_cust}.json")
        try:
            with open(dataset_path, "r") as f:
                self.dataset = json.load(f)
        except Exception:
            print(f"Warning: Could not load OSRM test dataset from {dataset_path}. Tests will use random data.")
            self.dataset = []
            
        self.n_problems = max(len(self.dataset), args.get('test_size', 256))
        self.reset()
        print(f"VRP DataGenerator ready. Train=random on-the-fly, Test={len(self.dataset)} OSRM instances.")

    def reset(self):
        self.count = 0

    def get_train_next(self):
        """Generate a random training batch with on-the-fly time matrices."""
        batch_size = self.args.get('batch_size', 128)
        
        input_pnt_batch = []
        demand_batch = []
        time_matrix_batch = []
        
        for _ in range(batch_size):
            # Random coordinates in [0,1] — same as original code
            coords = self.rnd.uniform(0, 1, size=(self.n_nodes, 2)).astype(np.float32)
            
            # Random demands for customers, depot=0
            demands = np.zeros(self.n_nodes, dtype=np.float32)
            demands[:self.n_cust] = self.rnd.randint(1, self.demand_max + 1, self.n_cust)
            
            # Compute time matrix on-the-fly from coordinates
            tm = compute_time_matrix(coords)
            
            input_pnt_batch.append(coords)
            demand_batch.append(demands)
            time_matrix_batch.append(tm)
                
        return (np.array(input_pnt_batch, dtype=np.float32), 
                np.array(demand_batch, dtype=np.float32), 
                np.array(time_matrix_batch, dtype=np.float32))

    def get_test_next(self):
        """Get next test instance from fixed OSRM dataset."""
        if self.dataset and self.count < len(self.dataset):
            inst = self.dataset[self.count]
            self.count += 1
            coords = np.array(inst["coordinates"], dtype=np.float32)
            demands = np.array(inst["demands"], dtype=np.float32)
            tm = np.array(inst["time_matrix"], dtype=np.float32)
            return (coords[np.newaxis], demands[np.newaxis], tm[np.newaxis])
        else:
            # Fallback: random test instance
            self.count = 0
            coords = self.rnd.uniform(0, 1, size=(self.n_nodes, 2)).astype(np.float32)
            demands = np.zeros(self.n_nodes, dtype=np.float32)
            demands[:self.n_cust] = self.rnd.randint(1, self.demand_max + 1, self.n_cust)
            tm = compute_time_matrix(coords)
            return (coords[np.newaxis], demands[np.newaxis], tm[np.newaxis])

    def get_test_all(self):
        """Get all test instances as a single batch."""
        test_size = self.args.get('test_size', 256)
        if self.dataset:
            instances = self.dataset[:test_size]
            coords = np.array([i["coordinates"] for i in instances], dtype=np.float32)
            demands = np.array([i["demands"] for i in instances], dtype=np.float32)
            tms = np.array([i["time_matrix"] for i in instances], dtype=np.float32)
            return (coords, demands, tms)
        else:
            # Fallback: generate random test batch
            coords = self.rnd.uniform(0, 1, size=(test_size, self.n_nodes, 2)).astype(np.float32)
            demands = np.zeros((test_size, self.n_nodes), dtype=np.float32)
            for i in range(test_size):
                demands[i, :self.n_cust] = self.rnd.randint(1, self.demand_max + 1, self.n_cust)
            tms = np.array([compute_time_matrix(c) for c in coords], dtype=np.float32)
            return (coords, demands, tms)


# ──────────────────────────────────────────────────────────────
# VRP Environment State
# ──────────────────────────────────────────────────────────────
class State(collections.namedtuple("State", ("load", "demand", "d_sat", "mask"))):
    pass
    
class Env(object):
    def __init__(self, args):
        self.capacity = args['capacity']
        self.n_nodes = args['n_nodes']
        self.n_cust = args['n_cust']
        self.input_dim = args['input_dim']
        
        # Placeholders
        self.input_pnt = tf.placeholder(tf.float32, shape=[None, self.n_nodes, 2])
        self.demand = tf.placeholder(tf.float32, shape=[None, self.n_nodes])
        self.time_matrix = tf.placeholder(tf.float32, shape=[None, self.n_nodes, self.n_nodes])
        
        self.batch_size = tf.shape(self.input_pnt)[0] 
        
    def reset(self, batch_size, beam_width=1):
        self.beam_width = tf.cast(beam_width, tf.int32)
        self.batch_beam = tf.cast(batch_size, tf.int32) * self.beam_width

        self.demand_tiled = tf.tile(self.demand, tf.stack([self.beam_width, 1]))
        self.load = tf.fill(tf.stack([self.batch_beam]), tf.cast(self.capacity, tf.float32))

        self.mask = tf.concat([
            tf.cast(tf.equal(self.demand_tiled, 0), tf.float32)[:, :-1],
            tf.ones(tf.stack([self.batch_beam, 1]))
        ], 1)

        return State(load=self.load, demand=self.demand_tiled, 
                     d_sat=tf.zeros(tf.stack([self.batch_beam, self.n_nodes])), mask=self.mask)

    def step(self, idx, batch_size, beam_parent=None):
        batch_size_int = tf.cast(batch_size, tf.int32)
        if beam_parent is not None:
            batch_tile = tf.reshape(self.beam_width, [1])
            batchBeamSeq = tf.expand_dims(tf.tile(tf.range(batch_size_int), batch_tile), 1)
            batchedBeamIdx = batchBeamSeq + batch_size_int * tf.cast(beam_parent, tf.int32)
            self.demand_tiled = tf.gather_nd(self.demand_tiled, batchedBeamIdx)
            self.load = tf.gather_nd(self.load, batchedBeamIdx)
            self.mask = tf.gather_nd(self.mask, batchedBeamIdx)

        BatchSequence = tf.expand_dims(tf.range(self.batch_beam), 1)
        batched_idx = tf.concat([BatchSequence, tf.cast(idx, tf.int32)], 1)

        demand_selected = tf.gather_nd(self.demand_tiled, batched_idx)
        
        new_load = self.load - demand_selected
        d_sat = tf.minimum(self.demand_tiled, tf.expand_dims(self.load, 1))

        self.demand_tiled = self.demand_tiled - \
                            tf.scatter_nd(batched_idx, demand_selected, tf.shape(self.demand_tiled))
                            
        self.load = tf.where(tf.equal(tf.squeeze(idx, 1), self.n_nodes - 1), 
                             tf.fill(tf.stack([self.batch_beam]), tf.cast(self.capacity, tf.float32)), 
                             new_load)

        self.mask = tf.concat([
            tf.cast(tf.equal(self.demand_tiled, 0), tf.float32)[:, :-1],
            tf.zeros(tf.stack([self.batch_beam, 1]))
        ], 1)

        self.mask = self.mask + tf.concat([
            tf.zeros(tf.stack([self.batch_beam, self.n_cust])),
            tf.cast(tf.equal(tf.reduce_sum(self.demand_tiled, 1), 0), tf.float32)[:, None]
        ], 1)

        return State(load=self.load, demand=self.demand_tiled, d_sat=d_sat, mask=self.mask)


# ──────────────────────────────────────────────────────────────
# Reward Function: OSRM Time Matrix Lookup
# ──────────────────────────────────────────────────────────────
def reward_func(idxs, time_matrix, env=None):
    """
    Computes total travel time by looking up step-by-step costs in the time_matrix.
    Route: depot -> idx[0] -> idx[1] -> ... -> idx[-1] -> depot
    """
    batch_size = tf.shape(time_matrix)[0]
    n_nodes = tf.shape(time_matrix)[1]
    total_time = tf.constant(0.0)
    
    batch_seq = tf.expand_dims(tf.range(batch_size), 1)
    depot_idx = tf.fill(tf.stack([batch_size, 1]), n_nodes - 1)
    
    if isinstance(idxs, list):
        for t in range(len(idxs)):
            idx_from = depot_idx if t == 0 else idxs[t-1]
            idx_to = idxs[t]
            gather_coords = tf.concat([batch_seq, tf.cast(idx_from, tf.int32), tf.cast(idx_to, tf.int32)], 1)
            total_time += tf.gather_nd(time_matrix, gather_coords)
            
        # Return to depot
        gather_final = tf.concat([batch_seq, tf.cast(idxs[-1], tf.int32), tf.cast(depot_idx, tf.int32)], 1)
        total_time += tf.gather_nd(time_matrix, gather_final)

    # CRITICAL: Penalize unserved demand heavily so the agent doesn't "cheat" by staying at the depot
    if env is not None:
        unserved_penalty = tf.reduce_sum(env.demand_tiled, 1) * 10000.0
        total_time += unserved_penalty

    return total_time
