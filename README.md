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

Here, the actor-critic method with deep neural networks (DNN) is used to stabilize the inverted pendulum.

Here, the DNN is designed such that policy and the value-function networks share some of the layers. This would allow faster training of the agents, presumably because the first layers of the DNNs extract features and map them to a more condensed representation space (a concept similar to transfer-learning). 

<img src="./statics/without-epsilon-layer.png" width="30%">

The main program is organized in the following way:
* **initialization**: random weights/biases are assigned to the network, 
* **experience loops**: 

  **(1)** a random initial state is assigned to the *state* variable,

  **(2)** given the *state*, an action (*a*) is chosen using the policy,

  **(3)**- the action *a* is given to the environment, and the environment returns the new state, the reward, and a signal indicating the end of the episode.
  
  **(4)**- if the process is not ended, the new state is assigned to the variable *state* and the execution continues to step **(2)** . 

All the states, actions, and the rewards are saved from the beginning of the episode until the end of it. This process is repeated for a number of episodes and all the data are gathered in an instance of the class *History*.

* **learning** : After sampling based on the policy, the obtained data is used to train the DNN. In the case our DNN, defining the loss function is not straightforward. The reason behind this complication is the fact that this DNN has two types of outputs (classification for the action and regression for the value function) which are both affected by the weight and biases of the *same* shared layers. To train the weights/biases of these shared layers one should combine the binary cross entropy loss for the actions, and the mean squared error for the value function. One way to combine these different losses would be to consider a (weighted) average of them. Using this loss and the data gathered from the experience, we used the actor-critic algorithm to solve make a policy iteration step. Using this new policy, we go back to the **experience loops**.


## results



## tips and tricks to stabilize it 
