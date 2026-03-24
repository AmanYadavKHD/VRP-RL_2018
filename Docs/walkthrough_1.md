# VRP-RL_2018 Code Modernization — Walkthrough

## Goal
Update the VRP-RL codebase (originally TensorFlow 1.x) to run on modern TF2 + Keras 3, configured for CPU with ~6GB RAM and 10 nodes.

## Changes Made

### 1. TF2 Compatibility Layer
All [.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/main.py) files had their imports updated:
```diff
-import tensorflow as tf
+import tensorflow.compat.v1 as tf
+tf.disable_v2_behavior()
```
**Files**: [main.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/main.py), [attention_agent.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py), [decode_step.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/shared/decode_step.py), [embeddings.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/shared/embeddings.py), [attention.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/shared/attention.py), [vrp_attention.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/VRP/vrp_attention.py), [vrp_utils.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/VRP/vrp_utils.py)

### 2. Replaced `tf.contrib` (Removed in TF2)
| Old API | Replacement |
|---|---|
| `tf.contrib.layers.xavier_initializer` | `tf.initializers.glorot_uniform` |
| `tf.contrib.rnn.BasicLSTMCell` | `tf.nn.rnn_cell.BasicLSTMCell` |

### 3. Replaced `tf.layers` with `tf.keras.layers` (Keras 3 Compat)
Keras 3 removed the `tf.layers` namespace. All layer calls updated:
```diff
-tf.layers.Conv1D(dim, 1, _scope=name)
+tf.keras.layers.Conv1D(dim, 1, name=name.replace('/', '_'))
```
Keras 3 also forbids `/` in layer names — added `.replace('/', '_')`.

**Files**: [embeddings.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/shared/embeddings.py), [attention.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/shared/attention.py), [vrp_attention.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/VRP/vrp_attention.py), [attention_agent.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py)

### 4. Forced Legacy Keras (Critical Fix)
`BasicLSTMCell`, `DropoutWrapper`, `MultiRNNCell` are still not in Keras 3. Installed `tf-keras` and set the env var before any TF import:
```python
os.environ['TF_USE_LEGACY_KERAS'] = '1'
```
**File**: [main.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/main.py) (line 3)

### 5. CPU / Low-Resource Configuration
In [configs.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/configs.py):
| Parameter | Old | New |
|---|---|---|
| `batch_size` | 128 | 32 |
| [n_train](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py#394-401) | 260000 | 10000 |
| `log_interval` | 500 | 50 |
| `gpu` | `"0"` | `""` (CPU) |

---

## Test Results

```
python main.py --n_train=2 --test_size=10 --n_nodes=11 --n_cust=10
```

```
Train Step: 0 -- Train reward: 7.324 -- Value: -0.246
  actor loss: -161.320 -- critic loss: 58.868
Average of greedy in batch-mode: 8.108 -- std 1.848
Average of beam_search in batch-mode: 6.960 -- std 1.509
Total time: 00:00:08
Exit code: 0 ✅
```

## How to Run

```bash
# Install dependencies
pip install tensorflow tqdm scipy numpy tf-keras

# Run with defaults (10 nodes, CPU, batch=32)
python main.py --task=vrp10

# Quick smoke test (2 steps)
python main.py --n_train=2 --test_size=10 --n_nodes=11 --n_cust=10
```
