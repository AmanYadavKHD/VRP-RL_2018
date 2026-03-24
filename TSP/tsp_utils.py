import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import collections
import json
import math

# ──────────────────────────────────────────────────────────────
# Utility: compute Euclidean-based surrogate time matrix
# ──────────────────────────────────────────────────────────────
def compute_time_matrix(coords, speed_factor=600.0):
    """Given [n_nodes x 2] coords in [0,1], return [n_nodes x n_nodes] time matrix (seconds)."""
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
        Phase 1 DataGenerator for TSP with Time Matrices.
        - Training: generates random coords each step (infinite diversity)
          + computes time matrix on-the-fly.
        - Testing: loads fixed OSRM JSON dataset from data/ folder.
        '''
        self.args = args
        self.rnd = np.random.RandomState(seed=args.get('random_seed', 123))
        self.n_nodes = args['n_nodes']
        
        # Load OSRM test dataset
        dataset_path = os.path.join("data", f"osrm_tsp{self.n_nodes}.json")
        try:
            with open(dataset_path, "r") as f:
                self.dataset = json.load(f)
        except Exception:
            print(f"Warning: Could not load OSRM test dataset from {dataset_path}. Tests will use random data.")
            self.dataset = []
            
        self.n_problems = max(len(self.dataset), args.get('test_size', 256))
        self.reset()
        print(f"TSP DataGenerator ready. Train=random on-the-fly, Test={len(self.dataset)} OSRM instances.")

    def reset(self):
        self.count = 0

    def get_train_next(self):
        """Generate a random training batch with on-the-fly time matrices."""
        batch_size = self.args.get('batch_size', 128)
        
        coords_batch = []
        tm_batch = []
        
        for _ in range(batch_size):
            coords = self.rnd.uniform(0, 1, size=(self.n_nodes, 2)).astype(np.float32)
            tm = compute_time_matrix(coords)
            coords_batch.append(coords)
            tm_batch.append(tm)
                
        return (np.array(coords_batch, dtype=np.float32), 
                np.array(tm_batch, dtype=np.float32))

    def get_test_next(self):
        """Get next test instance from fixed OSRM dataset."""
        if self.dataset and self.count < len(self.dataset):
            inst = self.dataset[self.count]
            self.count += 1
            coords = np.array(inst["coordinates"], dtype=np.float32)
            tm = np.array(inst["time_matrix"], dtype=np.float32)
            return (coords[np.newaxis], tm[np.newaxis])
        else:
            self.count = 0
            coords = self.rnd.uniform(0, 1, size=(self.n_nodes, 2)).astype(np.float32)
            tm = compute_time_matrix(coords)
            return (coords[np.newaxis], tm[np.newaxis])

    def get_test_all(self):
        """Get all test instances as a single batch."""
        test_size = self.args.get('test_size', 256)
        if self.dataset:
            instances = self.dataset[:test_size]
            coords = np.array([i["coordinates"] for i in instances], dtype=np.float32)
            tms = np.array([i["time_matrix"] for i in instances], dtype=np.float32)
            return (coords, tms)
        else:
            coords = self.rnd.uniform(0, 1, size=(test_size, self.n_nodes, 2)).astype(np.float32)
            tms = np.array([compute_time_matrix(c) for c in coords], dtype=np.float32)
            return (coords, tms)


# ──────────────────────────────────────────────────────────────
# TSP Environment State
# ──────────────────────────────────────────────────────────────
class State(collections.namedtuple("State", ("mask",))):
    pass

class Env(object):
    def __init__(self, args):
        self.n_nodes = args['n_nodes']
        self.input_dim = args['input_dim']
        
        self.input_pnt = tf.placeholder(tf.float32, shape=[None, self.n_nodes, args['input_dim']])
        self.time_matrix = tf.placeholder(tf.float32, shape=[None, self.n_nodes, self.n_nodes])
        
        self.batch_size = tf.shape(self.input_pnt)[0] 

    def reset(self, batch_size, beam_width=1):
        self.beam_width = tf.cast(beam_width, tf.int32)
        self.batch_beam = tf.cast(batch_size, tf.int32) * self.beam_width
        self.mask = tf.zeros(tf.stack([self.batch_beam, self.n_nodes]), dtype=tf.float32)
        return State(mask=self.mask)

    def step(self, idx, batch_size, beam_parent=None):
        batch_size_int = tf.cast(batch_size, tf.int32)
        if beam_parent is not None:
            batch_tile = tf.reshape(self.beam_width, [1])
            batchBeamSeq = tf.expand_dims(tf.tile(tf.range(batch_size_int), batch_tile), 1)
            batchedBeamIdx = batchBeamSeq + batch_size_int * tf.cast(beam_parent, tf.int32)
            self.mask = tf.gather_nd(self.mask, batchedBeamIdx)

        self.mask = self.mask + tf.one_hot(tf.squeeze(idx, 1), self.n_nodes)
        return State(mask=self.mask)


# ──────────────────────────────────────────────────────────────
# Reward Function: Time Matrix Lookup
# ──────────────────────────────────────────────────────────────
def reward_func(idxs, time_matrix):
    """
    Computes total travel time for a TSP tour by looking up in time_matrix.
    Tour: idx[0] -> idx[1] -> ... -> idx[-1] -> idx[0] (circular).
    """
    batch_size = tf.shape(time_matrix)[0]
    total_time = tf.constant(0.0)
    batch_seq = tf.expand_dims(tf.range(batch_size), 1)
    
    if isinstance(idxs, list):
        for t in range(len(idxs)):
            idx_from = idxs[t]
            idx_to = idxs[(t + 1) % len(idxs)]
            gather_coords = tf.concat([batch_seq, tf.cast(idx_from, tf.int32), tf.cast(idx_to, tf.int32)], 1)
            total_time += tf.gather_nd(time_matrix, gather_coords)
            
    return total_time