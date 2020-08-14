import numpy as np
import random
import gym as gym
from agent import Agent
from utilfunctions import initializer
from utilfunctions import one_hot
from utilfunctions import single_shape_adaptor
from utilfunctions import update_state_step
from rl_utils import Histories
from rl_utils import monitoring_performance


# create the environment
env = gym.make('CartPole-v0')

# creating the agent with the apropriate action and feature sizes
nr_features = env.observation_space.high.shape[0]
nr_actions = env.action_space.n

agent = Agent(nr_features=nr_features, nr_actions=nr_actions, gamma=0.98, epsilon=0.001, stddev=1, learning_rate=0.0025)

rounds_of_training = 1800
rendering_steps = 60# render the cart every rendering_steps iterations
dumping_steps = 60# for dumping the network
expr_per_learn = 60

# setting the random seeds
random.seed(1)
np.random.seed(3)
training_log = np.array([])
histories = Histories()

agent.model.save('./training-results/model-not-trained-agent')




for training_id in range(rounds_of_training):

    print("\nround: "+str(training_id))

    initial_state = env.reset()

    state, terminated, steps = initializer(initial_state)
    state = single_shape_adaptor(state, nr_features)

    while not terminated:

        action_id = agent.action_based_on_policy(state, env)

        one_hot_action = one_hot(action_id, nr_actions)

        new_state, reward, terminated, info = env.step(action_id)

        reward = reward / 1.0
        
        new_state = single_shape_adaptor(new_state, nr_features)
        
        histories.appending(reward, state, one_hot_action, terminated, env)

        state, steps = update_state_step(new_state, steps)
        if training_id % rendering_steps == 0:
            env.render()

    print("...    the terminal_state is reached after "+str(steps))

    if (training_id % expr_per_learn) == (expr_per_learn - 1):
        agent.learning(histories, env)
        histories = Histories()
    if training_id % dumping_steps == 0:
        print("...    saving the model")
        agent.model.save('./training-results/model-trained-agent', save_format='tf')

    training_log = monitoring_performance(training_log, training_id, steps, write_to_disk=True)

    

    print("round: "+str(training_id)+" is finished.")
