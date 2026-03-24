## 24-03-2026

Exactly What It Solves
This code solves the purest, classic mathematical variants of vehicle routing:

Classical 2D TSP: Navigating N randomly placed nodes in a 1x1 Euclidean grid with the shortest continuous loop.
Capacitated VRP (CVRP):
Single Depot: All routes start and end at the exact same location.
Static Coordinates: Customers are placed on a 2D Euclidean plane (straight-line distance).
Fixed Vehicle Capacity: A uniform capacity threshold (e.g., 20/30/40 units).
Discrete Demands: Customers need a static integer amount of goods (from 1 to 9).
Objective: Purely minimizing the total straight-line distance.
Exactly What It is NOT Solving (The Industry Gaps)
Industry logistics (like Amazon, FedEx, or Uber) practically never use the pure CVRP you just saw because it lacks physical world constraints. This codebase currently CANNOT solve:

No Time Windows (VRPTW): Customers do not have "Deliver between 9 AM and 12 PM" deadlines.
No Heterogeneous Fleets: It assumes you have infinite identical trucks. It cannot handle a mix of bikes, vans, and large trucks with different capacities and speeds.
No Dynamic / Stochastic Routing: All customers are known at Step 0 before the vehicle leaves. It cannot handle a new order arriving while the truck is already on the road.
No Real-World Maps: It uses straight-line $L_2$ Euclidean distance. It does not know about one-way streets, traffic, or asymmetric distances (where navigating A $\to$ B takes longer than B $\to$ A).
No Pickup-and-Delivery (PDVRP): It only "drops off". It cannot pick up an item from Customer A and deliver it to Customer B (like Uber or DoorDash).
What to Add for a High-Impact Industry Publication
If you want to modify this codebase for a massive publication that bridges Deep RL and real-world Operations Research, you should pick one of these major additions. (Traditional solvers like Gurobi or OR-Tools struggle with these when scaling, which is where RL shines).

1. Dynamic VRPTW (Vehicle Routing with Time Windows & Dynamic Orders) [Highest Industry Value]
The Idea: Modify the Env to track a "Clock" variable. Mask out customers whose time windows have expired. Randomly spawn new customers with new demands while the vehicle is currently executing the route.
Why it's publishable: Traditional algorithms take minutes to re-calculate a route when a new order drops. A trained RL agent computes the new best node in milliseconds. Demonstrating RL outperforming traditional solvers on dynamic routing is a massive, highly sought-after industry topic.
2. Makespan Optimization (Balancing Multiple Vehicles)
The Idea: Right now, the code minimizes the sum of all route distances. If one driver works 14 hours and another works 1 hour, the model doesn't care. Change the reward_func to minimize the Makespan (the duration of the longest single route), ensuring all drivers finish at roughly the same time.
Why it's publishable: Union rules and driver fatigue are massive real-world constraints. RL that natively balances workload equity across a fleet is a novel approach.
3. Real-World Map Embeddings (Graph Neural Networks)
The Idea: Rip out the 2D [x, y] continuous coordinates. Instead, pass a real graph representing city streets (using OpenStreetMap data) and use a Graph Neural Network (GNN) instead of LinearEmbedding.
Why it's publishable: Most RL VRP papers stop at 2D grids. Showing your RL agent learning to navigate real-world asymmetric street configurations proves it can actually be deployed to production.





##