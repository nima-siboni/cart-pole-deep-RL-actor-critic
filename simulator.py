import numpy as np
import os
import random
import gym as gym
from agent import Agent
from utilfunctions import initializer
from utilfunctions import one_hot
from utilfunctions import single_shape_adaptor
from utilfunctions import update_state_step
from rl_utils import Histories
from rl_utils import monitoring_performance

from tensorflow.keras.models import load_model

# create the environment
env = gym.make('CartPole-v0')

# creating the agent with the apropriate action and feature sizes
nr_features = env.observation_space.high.shape[0]
nr_actions = env.action_space.n

agent = Agent(nr_features=nr_features, nr_actions=nr_actions, gamma=0.98, epsilon=0.05, stddev=1, learning_rate=0.005)

# setting the random seeds
random.seed(1)
np.random.seed(3)
training_log = np.array([])
histories = Histories()

agent.model = load_model('./training-results/model-trained-agent')

figdir = './performance-and-animations/animations/trained/'

if not os.path.exists(figdir):
    os.makedirs(figdir)

initial_state = env.reset()

state, terminated, steps = initializer(initial_state)
state = single_shape_adaptor(state, nr_features)

while not terminated:

    action_id = agent.action_based_on_policy(state, env)

    one_hot_action = one_hot(action_id, nr_actions)

    new_state, reward, terminated, info = env.step(action_id)

    new_state = single_shape_adaptor(new_state, nr_features)

    histories.appending(reward, state, one_hot_action, terminated, env)

    state, steps = update_state_step(new_state, steps)
    img = env.render(mode='rgb_array')
    from PIL import Image

    img = Image.fromarray(img)
    if (steps < 10):
        timestamps = '00'+str(steps)
    else:
        if steps < 100:
            timestamps = '0'+str(steps)
        else:
            timestamps = str(steps)

    img.save(figdir+'state_'+timestamps+'.png')
