import numpy as np
from utilfunctions import scale_state

def calculate_the_A_hat(agent, histories, env):
    '''
    calculating the advantageto go hat{A} for each (state, action) in the history.
    
    Key arguments:
    histories -- the rewards for all the visited states
    gamma -- the discount factor

    output:
    the hat{A} which is of the same shape as reward_history.
    '''

    # prepare the s_{t+1}
    s = histories.scaled_state_history
    s_prime = np.roll(s, -1, axis=0)

    tmp = agent.model.predict(s_prime)[1] * (1 - histories.done_history)
    tmp = agent.gamma * tmp - agent.model.predict(s)[1]
    tmp = tmp.reshape(-1, 1)
    A_hat = histories.reward_history + tmp

    return A_hat

def cast_A_hat_to_action_form(A_hat, histories):

    output = np.multiply(histories.action_history, A_hat)

    return output


def monitoring_performance(log, training_id, steps, write_to_disk=True):
    '''
    returns a log (a numpy array) which has some analysis of the each round of training.

    Key arguments:

    training_id -- the id of the iteration which is just finished.
    steps -- the total number of steps before failing
    write_to_disk -- a flag for writting the performance to the disk

    Output:

    a numpy array with info about the iterations and the learning
    '''

    if training_id == 0:
        log = np.array([[training_id, steps]])
    else:
        log = np.append(log, np.array([[training_id, steps]]), axis=0)

    if write_to_disk:
        np.savetxt('steps_vs_iteration.dat', log)

    return log


class Histories():
    '''
    just a class to hold data
    '''
    def __init__(self):
        self.scaled_state_history = []
        self.reward_history = []
        self.action_history = []
        self.done_history = []
        
    def appending(self, reward, state, one_hot_action, done, env):

        scaled_state = scale_state(state, env)
        self.reward_history.append(reward)
        self.scaled_state_history.append(scaled_state)
        self.action_history.append(one_hot_action)
        self.done_history.append(done)
        

def preparing_the_V_target(agent, histories, env):
    '''
    Calculting the y as
    y_{t} = r(a_{t}, s_{t}) + gamma * V(s_{t+1})
    '''
    # create the y array with zeros
    # y = np.zeros_like(histories.reward_history)

    # y_{t} = r(a_{t}, s_{t})
    y = histories.reward_history + 0.0

    # prepare the s_{t+1}
    next_scaled_state = np.roll(histories.scaled_state_history, -1, axis=0)
    #next_scaled_state[-1, :] = next_scaled_state[-2, :] + 0
    tmp = agent.model.predict(next_scaled_state)[1] * (1 - histories.done_history)
    # tmp[-1, :] = np.zeros_like(tmp[-1, :])   

    # calculate V(s_{t+1})
    target = y + agent.gamma * tmp

    return target
