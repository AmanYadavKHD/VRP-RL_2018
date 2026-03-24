# From Euclidean Proximity to Real-World Time: Modernizing VRP-RL with Online Routing APIs (Phase 1)

## 1. Abstract & Motivation
Traditional Reinforcement Learning representations of the Vehicle Routing Problem (VRP) heavily rely on continuous 2D Euclidean planes to simulate constraints. While computationally convenient and theoretically profound, this spatial paradigm struggles when applied to commercial logistics. Real-world routes are bound by discrete road networks, asymmetric geographic barriers, and severe urban traffic topologies where spatial proximity (distance) does not reliably correlate with temporal cost (driving time). 

In Phase 1 of our architecture redesign, we dismantled the pure Euclidean foundation of Nazari et al. (2018). We transitioned the core attention mechanism and its continuous simulation environment into an **Online Real-World Routing Engine**. The entire execution, loss penalization, and policy gradients of the Neural Network are now completely driven by an explicit **Travel-Time Matrix** representing real-world seconds.

## 2. Hybrid Data Pipeline: OSRM Integration
Our primary engineering challenge was bridging the physical reality of a city with the sample-efficiency requirements of reinforcement learning.
- **The OSRM Evaluation Module:** For rigorous evaluation, we integrated with the Open Source Routing Machine (OSRM). Our `DataGenerator` generates bounding boxes over real cities, queries the API, and caches static JSON matrices containing the exact `[N x N]` driving times in seconds. Testing the model on this specific dataset guarantees performance validation on asymmetric physical topologies.
- **The Infinite-Diversity Training Surrogate:** Attempting to query an API 1,000,000 times during RL training would trigger instant IP bans. We resolved this by implementing an on-the-fly surrogate Euclidean-Time generator that mathematically converts random spatial uniform points into simulated driving times (e.g., `Euclidean Distance × 600 seconds`). This maintains infinite dataset diversity—preventing adversarial overfitting—while forcing the Attention mechanism to learn generalized temporal minimization.

## 3. Structural Stability: Resolving TensorFlow Rank Collisions
Our shift dynamically introduced discrete parameter shapes (such as `batch_size`), resulting in cascading `ValueError: Rank-0 tensors are not supported` errors and `ConcatOp` mismatches during inference and Beam Search. 
- Discarded unreliable `self.batch_size` inference references. We refactored `Env.reset()` and `Env.step()` to explicitly intercept dynamically calculated `batch_size` variables at runtime.
- We rigidly defined all dimension arrays as explicit Rank-1 vectors utilizing `tf.stack()`.
- Replaced ambiguous static shapes with `tf.shape()` operations, resolving the historic conflicts between Stochastic exploration arrays and Beam Search batch-widths.

## 4. The Reward Calculus: The Unserved Penalty Logic
Replacing the sum of spatial vectors with a discrete `gather_nd` index lookup introduced a hyper-logical loophole that the RL algorithm immediately attempted to exploit: 
- Because Euclidean distance from the Depot to the Depot is `0` (and equivalently, the OSRM time from Depot to Depot is `0`), the agent's optimal policy to minimizing travel time was to repeatedly select the Depot without ever serving any customers, scoring a "perfect" `0s` travel time over the decoded sequences.
- **The Fix:** We rewrote the `reward_func()` mathematically. During sequence finalization, the function sums the lingering elements in `env.demand_tiled`. For every unit of customer demand that is left unserved by the end of the sequence length, a severe multiplicative penalty `unserved_penalty = tf.reduce_sum(env.demand) * 10000.0` is permanently injected into the environment's `total_time` reward. The Actor must now exclusively focus on routing the highest density of customers in the shortest amount of physical time.

## 5. Telemetry & Visualization Overhaul
With the mathematical backend restructured into a strict temporal space, the visual infrastructure and logging parsers had to be deeply realigned. We executed a repository-wide terminology purge, rewriting all charts, log intervals, and `model_info.txt` outputs from Euclidean terms ("Tour Length", "Distance") toward empirical equivalents ("Avg Travel Time (seconds)", "Trip Time"). The visual inference plotter now tracks individual multi-stop trip durations per truck loop.

## 6. Conclusion
By officially replacing geographic proximity matrices with empirical time matrices bounded by a rigorous demand-penalization algorithm, Phase 1 successfully guarantees that our Attention Agent outputs financially viable routes perfectly aligned with industry Service-Level Agreements (SLAs). The agent is now structurally ready to ingest Phase 2: Time Windows.
