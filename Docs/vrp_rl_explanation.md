# VRP-RL_2018 Reverse Engineering Report

This report explains the implementation of the paper ["Reinforcement Learning for Solving the Vehicle Routing Problem"](https://arxiv.org/abs/1802.04240v2) based on the repository's source code. 

---

## 1. Repository Scan

The repository is structured to separate task-specific environments (TSP, VRP) from the core model architecture.

* **Complete Folder Structure:**
  * [TSP/](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/TSP/tsp_utils.py#7-48): Task-specific environment, data generation, and utilities for the Travelling Salesman Problem.
  * [VRP/](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/VRP/vrp_attention.py#3-76): Task-specific environment, data generation, and utilities for the Vehicle Routing Problem.
  * [model/](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py#307-311): The core RL model.
  * `shared/`: Shared components used by both TSP and VRP (e.g., embeddings, attention mechanisms, decoding steps).
  * [data/](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/VRP/vrp_utils.py#8-55): Directory where generated train/test datasets are saved.
* **Important Files:**
  * [main.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/main.py): **Entry point** for both training and inference.
  * [configs.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/configs.py): Handles argument parsing and sets up default hyperparameter configurations.
  * [task_specific_params.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/task_specific_params.py): Pre-defines task sizes (e.g., `vrp10`, `tsp50`), capacities, and dimensions.
  * [model/attention_agent.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py): Contains the [RLAgent](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py#7-399) class where the entire TensorFlow graph is built, including Actor and Critic networks, and loss computation.

---

## 2. Identify Execution Command

The command used to run the project for training VRP with 10 nodes is:
```bash
python main.py --task=vrp10
```
### What happens when the program starts:
1. [main.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/main.py) is the execution starting point.
2. It parses arguments via [ParseParams()](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/configs.py#28-108) which lives in [configs.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/configs.py) (and loads defaults from [task_specific_params.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/task_specific_params.py)).
3. It calls [load_task_specific_components(task_name)](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/main.py#13-37), which dynamically imports classes from the [TSP](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/TSP/tsp_utils.py#7-48) or [VRP](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/VRP/vrp_attention.py#3-76) folders based on the chosen task.
4. It instantiates [DataGenerator](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/TSP/tsp_utils.py#49-106), [Env](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/TSP/tsp_utils.py#110-166), and the core [RLAgent](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py#7-399).
5. It runs the main training loop `agent.run_train_step()` for [n_train](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py#392-399) steps.

---

## 3. Full Execution Flow

Here is the step-by-step pipeline from start to finish:
1. **Configuration Loading:** [configs.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/configs.py) reads `--task=vrp10` and fetches the constants for that task (nodes=11, capacity=20, etc.).
2. **Environment & Data Creation:** [DataGenerator](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/TSP/tsp_utils.py#49-106) starts creating random node distributions. [Env](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/TSP/tsp_utils.py#110-166) handles state representations, loads, and demands.
3. **Model Initialization:** [RLAgent](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py#7-399) builds the TensorFlow computation graph:
   * **Encoder Matrix:** `LinearEmbedding` embeds raw x,y coordinates.
   * **Decoder RNN:** Uses [RNNDecodeStep](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/shared/decode_step.py#130-235) to construct paths step-by-step using LSTM cells.
   * **Critic:** Evaluates the generated instances.
4. **Training Loop:** Over `args['n_train']` steps (default 260,000):
   * Queries a new batch from [DataGenerator](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/TSP/tsp_utils.py#49-106).
   * Calls `agent.run_train_step()` which executes a feed-forward pass, computes rewards, calculates Policy Gradient losses, applies Adam optimizers, and updates weights.
5. **Periodic Actions:** Every certain interval, it logs (`log_interval`), evaluates (`test_interval`), and checkpoints (`save_interval`).
6. **Evaluation:** When testing, `agent.inference()` executes either greedy or beam-search decoding over validation datasets.

---

## 4. Data Flow

Data moves efficiently through the batch-optimized pipeline:
* **Input Data Format:** Random coordinates and demands (e.g., shape `[batch_size, n_nodes, 3]` where 3 represents X, Y, and Demand). The depot's demand is 0.
* **Preprocessing:** `LinearEmbedding` maps raw dimensions to `embedding_dim` (128 by default).
* **Batch Generation:** Generated on the fly in `DataGenerator.get_train_next()`. The default batch size is 128.
* **Flow into the Model:** 
  * The environment state includes tracking remaining capacities ([load](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py#307-311)), unfulfilled `demand`, and a `mask`.
  * The encoder creates `encoder_emb_inp`.
  * The decoder sequentially generates actions. At each step $t$, the previously selected node and the context vectors are fed into the LSTM state. The attention mechanisms output a probability distribution over the available nodes.
* **Model Outputs:** The actor outputs the sequence of node indices (the route). The environment interprets this sequence, updates its state, and eventually produces a negative route length as the reward `R`, while the Critic outputs the baseline baseline prediction `v`.

---

## 5. Model Architecture

The model uses a Sequence-to-Sequence (Seq2Seq) framework located mostly in [model/attention_agent.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py) and [shared/decode_step.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/shared/decode_step.py):
* **Encoder:** It isn't a complex RNN. Instead, coordinates are simply linearly embedded using [shared/embeddings.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/shared/embeddings.py) (`LinearEmbedding`).
* **Decoder:** Designed with [RNNDecodeStep](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/shared/decode_step.py#130-235), featuring multi-layer LSTM cells. It maintains the current state of decoding.
* **State & Action Representation:** The state includes the decoder's hidden layer and the context from the encoder, enriched by VRP properties (vehicle load, node demand) fed via [AttentionVRPActor](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/VRP/vrp_attention.py#3-76). The action is a single node index (the next destination).
* **Attention Mechanism:** Implemented as a Pointer Network ([shared/attention.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/shared/attention.py)). It calculates "glimpses" over the unvisited nodes. For VRP, the custom [AttentionVRPActor](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/VRP/vrp_attention.py#3-76) (in [VRP/vrp_attention.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/VRP/vrp_attention.py)) integrates dynamic context such as `demand` and [load](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py#307-311) alongside geographic features before computing the query-reference affinities.

---

## 6. Algorithm Implementation

The optimization utilizes **Reinforcement Learning** (specifically, **REINFORCE with a Baseline**):
* **RL Method:** Policy Gradient. The model is an actor-critic where the Critic reduces variance by calculating a baseline.
* **Reward Calculation:** Handled by [reward_func](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/VRP/vrp_utils.py#252-289) (in [TSP/tsp_utils.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/TSP/tsp_utils.py) or [VRP/vrp_utils.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/VRP/vrp_utils.py)). It calculates the Euclidean distance traversed by the output sequence and returns it as a negative scalar (minimize distance = maximize reward).
* **Loss Functions (in [build_train_step](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py#252-301)):**
  * **Actor Loss:** `mean((R - v) * logprobs)` where `R` is reward, `v` is critic baseline, and `logprobs` is the logarithmic probability of the chosen sequence.
  * **Critic Loss:** Mean Squared Error: `MSE(R, v)`.
* **Gradient Update:** Uses Adam Optimizer (`actor_net_lr` and `critic_net_lr` = 1e-4), limits explosions using `max_grad_norm` clipping.

---

## 7. Training Pipeline

* **Dataset Generation:** Generated fully dynamically during training (`rnd.uniform(0,1)` for coordinates and `rnd.randint(1,10)` for demands).
* **Batch Size:** 128 (configurable).
* **Epochs/Steps:** No fixed epochs. Simply loops over [n_train](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py#392-399) steps (default 260,000 steps).
* **Learning Rate:** 1e-4 for both networks.
* **Checkpoints:** Handled by `tf.train.Saver` every `save_interval` (default 10,000) inside the model directory.
* **Logging:** Prints loss, average rewards, and timings to stdout and a `results.txt` file inside `./logs/task_date_time/`.

---

## 8. Evaluation Pipeline

Tested within the same script, but with graph nodes switched to different decoding states.
* **Evaluation Scripts:** Handled primarily by [evaluate_batch()](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py#361-382) and [evaluate_single()](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py#312-359) directly inside [RLAgent](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py#7-399).
* **Decoding Modes:** Uses either `greedy` (always plucking the highest probability node) or `beam_search` (maintaining the `n` best active paths to avoid greedy traps). The beam width defaults to 10.
* **Metrics:** Overall average reward (tour length) and the standard deviation on a pre-generated separate "test dataset" of 1,000 instances.

---

## 9. File-by-File Explanation

* **[main.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/main.py):** Initiates the runtime flow. Parses configs, spins up the Environment and Agent, manages the train/test iterations, logs outputs, and manages checkpoints.
* **[configs.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/configs.py):** Argument parser for hyperparameters (dropout, capacities, lr, layers, sizes). Creates timestamped directories.
* **[task_specific_params.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/task_specific_params.py):** Contains quick structural definitions for baseline tasks (like `tsp10`, `tsp50`, `vrp20`, `vrp100`) regarding payload sizes and max array lengths.
* **[model/attention_agent.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py):** The heart of the implementation. Defines [RLAgent](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py#7-399). Contains the primary graph mappings (Actor/Critic initializations, TensorFlow loop mappings via Beam Decoding vs Greedy, gradients mapping, [build_train_step](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py#252-301)).
* **[shared/decode_step.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/shared/decode_step.py):** Contains [DecodeStep](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/shared/decode_step.py#3-129) and [RNNDecodeStep](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/shared/decode_step.py#130-235). Implements the step-by-step loop for sequence generation managing LSTM states and executing glimpse processes.
* **[shared/attention.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/shared/attention.py):** Contains the Base [Attention](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/shared/attention.py#3-50) Pointer network used for predicting the log-probabilities of subsequent nodes.
* **[VRP/vrp_utils.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/VRP/vrp_utils.py) / [TSP/tsp_utils.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/TSP/tsp_utils.py):** Contains the [DataGenerator](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/TSP/tsp_utils.py#49-106), [Env](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/TSP/tsp_utils.py#110-166), and [reward_func](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/VRP/vrp_utils.py#252-289). Notably, the `Env.step` function is extremely important as it dynamically masks nodes: for instance, masking full customers, masking if the car is out of capacity, and handling depot returns.
* **[VRP/vrp_attention.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/VRP/vrp_attention.py):** Overrides standard attention with [AttentionVRPActor](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/VRP/vrp_attention.py#3-76)/[Critic](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/VRP/vrp_attention.py#78-149) to dynamically factor `env.demand` and `env.load` arrays directly into the neural representation calculations via 1D Convolutions.

---

## 10. Simplified Explanation

Think of this code as a **game playing robot** trying to become a very efficient delivery driver. 

1. **The Game Board (Environment & Data Generator):** Every round, 128 random maps are generated. Each map has a depot and random customers requesting varying numbers of packages.
2. **The Rules (Masking):** The vehicle can only hold so much (`capacity`). It cannot visit a customer that wants 5 packages if it only has 3 left. It cannot visit customers who already got their packages. The environment enforces these rules by "Masking" those choices out, hiding them from the robot's options.
3. **The Brain (Actor Decoder & Attention):** The robot is built on an LSTM network (which holds memory of what it just did). Every time it pulls up to an intersection, it calculates "attention"—a spotlight assessing all available remaining nodes, factoring in geographic location and its payload weight versus the customer's demands. It outputs probabilities and chooses the likeliest route.
4. **The Scorekeeper (Critic & Reward):** At the very end of the route, the system measures the total distance driven. The Critic tries to guess how long the route was going to be. If the robot's route was much shorter (better) than the Critic expected, the robot gets heavily rewarded. If it was longer, it implies the choices were poor. The network adjusts its internal wiring (Gradients) and tries again. over 260,000 times!

### For your modifications:
* **To change the model architecture:** Look at [model/attention_agent.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/model/attention_agent.py) and [shared/decode_step.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/shared/decode_step.py).
* **To add rules/constraints:** You must alter `Env.step()` in [VRP/vrp_utils.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/VRP/vrp_utils.py) to change the `mask`.
* **To test your own benchmarks:** Modify [DataGenerator](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/TSP/tsp_utils.py#49-106) in [VRP/vrp_utils.py](file:///c:/COLLEGE/SEM%20X/WORK/VRP-RL_2018/VRP/vrp_utils.py) to load standard datasets (like CVRPLIB) instead of `rnd.uniform`.





###
1. REINFORCE (The Classic Approach)
How it works: It uses two separate neural networks. The Actor builds the route. A completely separate neural network (the Critic) looks at the exact same starting map and tries to guess $V$ (the expected distance for this problem).
The Math: Advantage = $R - V$ (Actual distance vs. Critic's guess).
Update: The Actor is told to increase the probability of that route proportional to $(R - V)$. The Critic is updated using Mean Squared Error to make a more accurate guess next time.
2. Advantage Actor-Critic (A2C)
How it works: A2C is conceptually almost identical to REINFORCE. It uses the exact same Actor network and the exact same Critic network.
The Difference: The big problem with REINFORCE is that the Actor often finds a "mediocre" route early in training and just memorizes it perfectly, refusing to try new things. A2C fixes this by adding an Entropy Bonus to the Actor's loss. Entropy is mathematical randomness. The loss function actively rewards the Actor for keeping its probabilities slightly random, forcing it to "explore" alternative routes throughout training.
3. PPO (Proximal Policy Optimization)
How it works: PPO solves a different problem with REINFORCE: catastrophic jumps. If the Actor accidentally finds one exceptionally amazing route, standard REINFORCE will violently update the neural network weights to prioritize that route. This violent update often destroys everything else the network had learned, causing the training to collapse.
The Difference: PPO calculates the same $R - V$ Advantage. But before applying the update, it calculates the ratio of the "New Probability" / "Old Probability". It then mathematically clips this ratio between [0.8, 1.2]. This guarantees that the neural network can never change its routing behavior by more than ~20% in a single step, no matter how good the reward was. It results in extremely smooth, stable learning.
4. Greedy Rollout Baseline (POMO/Kool et al.)
How it works: This is the most brilliant and modern algorithm for routing problems (2019+). The authors realized: Why are we wasting processing power training a secondary Critic neural network just to guess a Baseline $V$? Training two networks at once is unstable.
The Difference: Instead of a Critic network, the Greedy Baseline simply runs the Actor network twice.
First, it runs the Actor stochastically (picking customers based on probability) to get Reward $R$.
Second, it runs the exact same Actor greedily (always picking the #1 highest probability customer) to get $R_{greedy}$. The baseline $V$ is simply $R_{greedy}$.
Update: If the stochastic route was better than the greedy route ($R > R_{greedy}$), the Actor learns that exploring that specific path was a good idea and updates its weights. No Critic network required!





###
1. Multiple Vehicles (Heterogeneous Fleet VRP)
Right now, your code effectively already handles "multiple vehicles", but it assumes they are all identical. A "vehicle" is just a loop from the Depot back to the Depot. If you wanted a fleet of vehicles with different sizes (e.g., a truck with capacity 50 and two vans with capacity 20):

The Change: You would modify 

VRP/vrp_utils.py
. The 

State
 tuple currently only tracks a single scalar 

load
. You would need to change it to track an array like [load_truck, load_van1, load_van2].
The Architecture: The Decoder's "Attention Mechanism" would need to make two decisions at every step: 1) Which customer to visit next, and 2) Which specific vehicle to send.
2. Time Windows (VRPTW)
This is the single most common extension to VRP (e.g., "Deliver this package between 9:00 AM and 11:00 AM"). To add Time Windows to your code:

The Input: Currently, your input_dim = 3 (X, Y, Demand). You would change input_dim = 5 to represent (X, Y, Demand, Ready_Time, Due_Time).
The State Tracking: In 

vrp_utils.py
, your environment would need to track current_time. Every time the agent takes a step, current_time = current_time + travel_distance + service_time.
The Mask: The neural network mathematically cannot pick invalid customers because of self.mask. You would simply add a line of code to the mask logic: If current_time + travel_distance > Due_Time for Customer 4, heavily mask Customer 4 so the AI is physically forbidden from selecting it.
3. Traffic Conditions (Time-Dependent VRP)
In your current code, the distance between two nodes is purely a straight line on a graph (Euclidean distance $c^2 = a^2 + b^2$). In real life, traveling 10 miles at 2:00 AM is much faster than 10 miles at 5:00 PM. To add Traffic:

The Reward Function: The most critical change would be in 

reward_func()
 inside 

vrp_utils.py
. Right now, it penalizes the AI purely based on straight-line mathematical distance.
The Code: You would replace the Euclidean distance equation (tf.pow(..., .5)) with a function that calculates time penalty based on a simulated "Traffic Matrix" or time-of-day multiplier.
Why RL is the Future for These Problems
If you add Traffic and Time Windows to a mathematical solver like Gurobi or CPLEX, the complexity explodes exponentially. A 50-node problem with time windows can take massive server clusters hours to solve.

With your Reinforcement Learning architecture, adding Time Windows just means adding 2 extra numbers to the input layer and modifying the mask logic! The neural network will still spit out a highly optimized route in milliseconds. This speed difference is arguably the most powerful point you can make in defense of your thesis.



##
1. Create a "Hybrid AI" (Easy)

What to do: Write a basic Python 2-opt function in 

view_routes.py
. It takes the route the AI generates and surgically uncrosses any overlapping lines.
Why it's modern: Industry logistics (like FedEx/Amazon) never use pure AI. They all use Hybrid Solvers (AI for speed + classic algorithms to clean up the math).
2. Add Time Windows / Traffic (Medium)

What to do: Modify the environment (

VRP/vrp_utils.py
) to give customers "due times". If the driver would arrive late, mathematically forbid the AI from picking that customer.
Why it's modern: Standard VRP is a toy problem. Adding "Time Windows" proves the AI can handle brutal real-world constraints.
3. Multi-Agent Reinforcement Learning (Medium)

What to do: Give each vehicle its own independent Actor network. Let them "negotiate" over which customers to take.
Why it's modern: This simulates real-world decentralized ride-sharing and dynamic dispatching (Uber/Lyft).
4. Upgrade to Transformers (Hard / Cutting-Edge)

What to do: Replace the 2018 LSTM code in 

model/attention_agent.py
 with the 2021 Multi-Head Self-Attention Transformer architecture.
Why it's modern: It trains infinitely faster, solves 100+ node problems better, and proves you understand modern LLM infrastructure!




###
running analysis - 
python view_routes.py --is_train=False --task=vrp20 --n_show=20 --load_path "logs\vrp20-2026-03-14_11-45-01\model\model.ckpt-29999"

python analyze_results.py --log_dir "logs\vrp20-2026-03-14_11-45-01"






###
Based on the environment logic and data generation inside the codebase (

VRP/vrp_utils.py
, 

TSP/tsp_utils.py
, and 

task_specific_params.py
), this repository is solving two very specific, classic variants of routing problems:

1. The TSP (Traveling Salesperson Problem)
It is solving the standard 2D Euclidean TSP.

Nodes: 10, 20, 50, or 100 nodes are randomly generated in a continuous 2D space (with X and Y coordinates between 0.0 and 1.0).
Objective: Find the shortest possible continuous route (using Euclidean/straight-line distance) that visits exactly every node once and returns to the starting node.
No constraints: There are no capacities or demands, just pure distance minimization.
2. The VRP (Vehicle Routing Problem)
It is solving the classic Capacitated Vehicle Routing Problem (CVRP). Specifically, it is a Single-Depot, 2D Euclidean CVRP.

Here are the strict rules it plays by:

One Single Depot: The very last node in the data array is hardcoded as the depot. It always has 0 demand.
Stochastic 2D Locations: The 10, 20, 50, or 100 customers are randomly scattered in the same 0.0 to 1.0 2D grid.
Discrete Customer Demands: Each customer is randomly assigned a single integer demand between 1 and 9 (e.g., they need 4 boxes).
Homogeneous Fixed Capacity: The vehicle has a single strict maximum capacity (e.g., 20 units of load for VRP10).
Refill Mechanics: When the vehicle's remaining load is less than the next customer's demand, it must return to the depot to magnetically refill to its max capacity.
Objective: Minimize the total Euclidean distance travelled by the vehicle to satisfy all demands and end its shift parked at the depot.
What it is NOT solving: It is not solving more complicated variants. There are:

No Time Windows (CVRPTW) (Customers don't have deadlines).
No Multiple Depots (MDVRP).
No Pickups and Deliveries (VRPPD) (The vehicle only delivers from the depot; it doesn't pick up items from customer A and take them to customer B).
No Dynamic Routing (All customers and their demands are known before the vehicle leaves the depot)