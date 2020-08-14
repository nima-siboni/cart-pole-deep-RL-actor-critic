import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from rl_utils import preparing_the_V_target
from utilfunctions import reshaping_the_histories, scale_state
from rl_utils import calculate_the_A_hat
from rl_utils import cast_A_hat_to_action_form
from rl_utils import preparing_the_V_target

class Agent:
    '''
    the agent class which has the policy.
    - takes actions based on the policy
    - 
    '''
    def __init__(self, nr_features, nr_actions, gamma=0.99, epsilon=0.02, stddev=0.2, learning_rate=0.01):

        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=stddev, seed=1)
        optimzer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        inputs = keras.layers.Input(shape=(nr_features))
        x = layers.Dense(256, activation='relu', kernel_initializer=initializer)(inputs)
#        x = layers.Dense(256, activation='relu', kernel_initializer=initializer)(x)

        output_policy = layers.Dense(nr_actions, activation='softmax', kernel_initializer=initializer, name = 'output_policy')(x)
        output_policy = (output_policy + epsilon) / (1.0 + epsilon * nr_actions)

        output_V = layers.Dense(1, activation='linear', kernel_initializer=initializer, name='output_V')(x)

        self.model = keras.Model(inputs=inputs, outputs=[output_policy, output_V])

        self.model.compile(optimizer='adam',
                           loss={'tf_op_layer_RealDiv': 'categorical_crossentropy', 'output_V': 'mse'}),
                           #metrics=['categorical_crossentropy', 'mse'])

        #initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=stddev, seed=1)
        #optimzer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        #inputs_V = keras.layers.Input(shape=(nr_features))
        #x_V = layers.Dense(256, activation='relu', kernel_initializer=initializer)(inputs_V)
        #x_V = layers.Dense(256, activation='linear', kernel_initializer=initializer)(x_V)
        #self.V = keras.Model(inputs=inputs_V, outputs=output_V)
        #self.V.compile(optimizer='adam', loss="mse", metrics=["accuracy"])
        
        self.gamma = gamma

    def action_based_on_policy(self, state, env):
        '''
        Returns the chosen action id using the policy for the given state
        '''
        scaled_state = scale_state(state, env)
        probabilities = self.model.predict(scaled_state)[0][0]
        #print(state, probabilities)
        nr_actions = len(probabilities)
        chosen_act = np.random.choice(nr_actions, p=probabilities)
        return chosen_act

    def learning(self, histories, env):
        '''
        the learning happens here:
        1- first the traingin data for value function and policy are calculated,
        2- the model is trained.
        '''
        ## TODO: Here the model is called two times for the same tmp_histories
        ## once for the V (1.1) and once for the policy (1.2). This is redundent
        
        print("...    reshaping the data")
        tmp_histories = reshaping_the_histories(histories, env)
        
        # 1.1 preparing the target value for traingin V
        print("...    preparing target for V-calculation")
        target_for_V_training = preparing_the_V_target(self, tmp_histories, env)

        # 1.2 calculating the advantages
        print("...    preparing target for policy-calculation")
        A_hat = calculate_the_A_hat(self, tmp_histories, env)
        A_hat = cast_A_hat_to_action_form(A_hat, tmp_histories)
        
        # 2.2 fitting the model
        #callbacks = [keras.callbacks.TensorBoard(log_dir='./logs')]
        print("...    training the policy/V")
        print("...        with sample size of ", np.shape(tmp_histories.scaled_state_history)[0])
        fitting_log = self.model.fit(x=tmp_histories.scaled_state_history,
                                      y={'tf_op_layer_RealDiv': A_hat, 'output_V':target_for_V_training},
                                      epochs=1, verbose=1)
