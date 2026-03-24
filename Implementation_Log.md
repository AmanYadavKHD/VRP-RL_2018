# Core Codebase Implementation Log

**Objective:** Upgrading the pure 2D Euclidean CVRP / TSP codebase into an Online Real-World Routing Engine.

---

## [Phase 1] Real-World Time Matrices (OSRM API Integration)
**Status: IN PROGRESS — Code Complete, Pending Training Verification**

### Architecture Decision: Training vs Testing Data Flow

| Component | Original Code | Phase 1 Code |
|-----------|-------------|-------------|
| **Training Data** | Random `[0,1]` coords each step (infinite diversity) | Random `[0,1]` coords each step + **on-the-fly time matrix** via `Euclidean × 600` |
| **Testing Data** | Fixed `.txt` files in `data/` | Fixed OSRM JSON files in `data/` (real driving times from API) |
| **Reward Signal** | `sqrt(dx² + dy²)` Euclidean distance | `tf.gather_nd` lookup into time matrix (seconds) |

**Key Insight:** Training still uses infinite random diversity (no overfitting risk). The model learns time-optimal routing because the gradient signal comes from time-based rewards. Testing on real OSRM data proves generalization to physical road networks.

---

### 1. Data Generation (`osrm_data_generator.py`)
* **Upgraded** with CLI arguments: `--task`, `--n`, `--mode`, `--seed`
* **Two modes:**
  * `--mode osrm`: Fetches real driving times from OSRM public API (NYC GPS bounds). Normalizes GPS coords to `[0,1]` for neural network.
  * `--mode fallback`: Fast Euclidean surrogate for offline use.
* **Usage:** `python osrm_data_generator.py --task vrp10 --n 50 --mode osrm`

### 2. VRP Environment (`VRP/vrp_utils.py`) — Full Rewrite
* **`DataGenerator`:**
  * `get_train_next()`: Generates random coords + demands + computes time matrix on-the-fly each step. Infinite training diversity.
  * `get_test_next()` / `get_test_all()`: Loads fixed OSRM JSON from `data/osrm_vrp{n_cust}.json`.
* **`Env` class:**
  * Added `self.time_matrix = tf.placeholder(shape=[None, n_nodes, n_nodes])`.
  * `reset(batch_size, beam_width)`: Now takes explicit `batch_size` tensor to avoid cross-graph shape leakage.
  * `step(idx, batch_size, beam_parent)`: Uses passed `batch_size` for beam-parent reconstruction instead of `self.batch_size` placeholder shape.
  * All `tf.zeros`, `tf.fill`, `tf.ones` calls use `tf.stack()` for Rank-1 shape vectors.
* **`reward_func(idxs, time_matrix)`:** Pure `tf.gather_nd` lookup: `depot → idx[0] → idx[1] → ... → depot`. Returns total travel time in seconds.

### 3. TSP Environment (`TSP/tsp_utils.py`) — Full Rewrite
* Same architecture as VRP: random training, OSRM testing.
* `Env.reset(batch_size, beam_width)` and `Env.step(idx, batch_size, beam_parent)` aligned with VRP API.
* `reward_func`: Circular tour lookup `idx[0] → idx[1] → ... → idx[-1] → idx[0]`.

### 4. VRP Attention Mechanism (`VRP/vrp_attention.py`)
* **Actor (line 41):** Fixed `env.demand` → `env.demand_tiled` (reads live demand that updates during decode steps, not the raw placeholder)
* **Critic (line 121):** Fixed `env.input_pnt[:,:,-1]` → `env.demand` (demand is now a separate placeholder, not the 3rd column of input_pnt which only has `[x,y]`)

### 5. Attention Agent (`model/attention_agent.py`)
* **Re-enabled** greedy and beam_search model builds (were disabled during debugging)
* **Removed** all `DEBUG: ...` print statements
* **Removed** stale `env.reset()` call in `evaluate_batch()` (reset now happens inside `build_model`)
* **Feed dicts** in `run_train_step()`, `evaluate_single()`, `evaluate_batch()` correctly unpack `time_matrix` alongside `input_pnt` and `demand`

### 6. Training Script (`main.py`)
* **Re-enabled** `agent.inference()` call after each log interval

---

### Bug Fixes Summary (TensorFlow 1.x Graph Stability)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `ValueError: Rank-0 tensors not supported` | `tf.zeros([scalar, n])` with scalar tensor in list | Changed to `tf.zeros(tf.stack([tensor, n]))` |
| `TypeError: int32 vs int64` | Mixed index types in `tf.concat`, `tf.gather_nd` | Standardized all indices to `tf.int32` |
| `Incompatible shapes [50,11] vs [32,11]` | `Env.step()` used `self.batch_size` (placeholder dim) instead of passed batch_size | Changed to use explicit `batch_size` parameter |
| `AttributeError: input_data` | Critic accessed wrong attribute | Changed `input_data` → `input_pnt` |
| Cross-graph contamination | Shared `env` object mutated by multiple `build_model()` calls | Each build gets fresh `env.reset(batch_size, beam_width)` |

---

## [Phase 2] Time Windows (VRPTW)
**Status: PENDING**
*(Modifications will be logged here as they are written...)*
