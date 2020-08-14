# cart-pole-deep-RL-actor-critic
Solving the inverted pendulum problem with deep-RL actor-critic (with shared network between the value-evaluation and the policy, epsilon-greedy policy). Some implementation issues concerning the stability are discussed. 

## the problem

The problem is to train an agent which stablizes a pole on a cart (an inverted pendulum problem). This is a classic problem in dynamics and control theory and is used as a benchmark for testing control strategies [[1]](https://en.wikipedia.org/wiki/Inverted_pendulum#:~:text=An%20inverted%20pendulum%20is%20a,additional%20help%20will%20fall%20over).

In the current presentation of the problem, the agent:
- observes the *current* position and velocity of the cart and the pole, 
- can only move the cart forward and backward for a fixed distance (+1/-1) every time. So the action space is discrete.


 
The environment is cart-pole-v1 env. from OpenAI Gym. A deterministic environment whicg rewards the agent by +1 every time step that it does not fail. The failing is defined as The agent fails when the angle between the pole and the vertical line exceeds a certain threshold. Here is an example of failure when the controller just moves the cart irrespective of the state of the pole.


<img src="./performance-and-animations/animations/not-trained/animation.gif" width="60%">


## the approach

Here I have implemented the actor-critic method with deep neural networks (DNN).

Here, the DNN is designed such that policy and the value-function networks share some of the layers. This would allow faster training of the agents, presumably because the first layers of the DNNs extract features and map them to a more condensed representation space (a concept similar to transfer-learning). 


## results

## tips and tricks to stabilize it 
