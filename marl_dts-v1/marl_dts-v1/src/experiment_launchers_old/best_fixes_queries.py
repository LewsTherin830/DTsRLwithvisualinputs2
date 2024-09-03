import os
import sys
import gym
from time import time
import random
import numpy as np
from copy import deepcopy
sys.path.append(".")
import utils
from algorithms import genetic_programming
from algorithms import continuous_optimization
from decisiontreelibrary import QLearningLeafFactory, ConditionFactory, \
        RLDecisionTree
from FrameStackingWrapper import FrameStackingWrapper
import cv2
from matplotlib import pyplot as plt


def find_indexes_2d(array, value):
    indexes = []
    for i, row in enumerate(array):
        for j, element in enumerate(row):
            if element == value:
                indexes.append((i, j))
    if(len(indexes) > 0):
        indexes = np.array(indexes)
        return np.mean(indexes, axis = 0)
    else:
        return [0,0]

def display_numpy_array(array, delay=0.001):
    
    
    array = array[:,:,0]
    
    plt.imshow(array)
    arr = find_indexes_2d(array, 213)
    print(arr)
    plt.plot(arr[1],arr[0], 'ro')
    arr = find_indexes_2d(array, 236)
    plt.plot(arr[1],arr[0], 'ro')
    arr = find_indexes_2d(array, 92)
    plt.plot(arr[1],arr[0], 'ro')
    plt.show(block=False)  # Show plot without blocking
    plt.pause(delay)       # Pause for 'delay' seconds
    #plt.close()            # Close the plot

def observe(observation):
    
    #preprocessing observation
    observation = observation[34:194,:,:] #cropping unnecessary parts
    #observation = np.resize(observation,(80,80,3))
    res = cv2.resize(observation, dsize=(40, 40), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)
    
    display_numpy_array(observation)
    return observation
    

env = gym.make("PongNoFrameskip-v4")

# Iterate over the episodes
for i in range(1):
    env.seed(i)
    obs = env.reset()
    done = False

    while not done:
        obs2 = observe(obs)
        #print(obs)
        obs, rew, done, _ = env.step(0)
