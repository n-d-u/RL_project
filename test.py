from cupshelpers import activateNewPrinter
import gym 
import random
# !ale-import-roms roms/
from rl.policy import LinearAnnealedPolicy
from rl.agents import DQNAgent
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

game = 'SpaceInvaders-v0'
environment = gym.make(game, render_mode = "human")
h, w, ch = environment.observation_space.shape

CNN = Sequential()
CNN.add(Conv2D(32,(8,8), input_shape = (3,h, w, ch), strides = (4,4), activation = 'relu'))
CNN.add(Conv2D(64,(4,4), strides = (2,2), activation = 'relu'))
CNN.add(Conv2D(64,(3,3), strides = (1,1), activation = 'relu'))
CNN.add(Flatten())

CNN.add(Dense(512,activation='relu'))
CNN.add(Dense(256,activation='relu'))
CNN.add(Dense(environment.action_space.n, activation = 'linear'))



policy1 = EpsGreedyQPolicy()
agent = DQNAgent(model = CNN, memory = SequentialMemory(limit=1000, window_length = 3), 
                    policy = LinearAnnealedPolicy(policy1, attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000), 
                    enable_dueling_network=True, dueling_type='avg', 
                    nb_actions=environment.action_space.n, nb_steps_warmup=1000 )


agent.compile(Adam(lr=1e-4))

agent.load_weights('Weights/dqn_weights.h5')

scores = agent.test(environment, nb_episodes=10, visualize=False)
print(scores.history['episode_reward'])

