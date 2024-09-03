#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    experiment_launchers.pz_magent_launcher
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This module allows to launch experiments for the MAgent environments
    in PettingZoo.
    This module only manages the case in which we want to evolve only a team
    against random opponents.
    Not suitable for non-teamed environments such as gather, since the fitness
    function cannot give a measure of goodness in such contexts.

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
    
    
    runfile('C:/Users/Kirito/Desktop/marl_dts-master-src/marl_dts-master-src/src/experiment_launchers/pong_gp_no_queries.py', args="C:/Users/Kirito/Desktop/marl_dts-master-src/marl_dts-master-src/src/experiment_launchers/configs/pong_attention_gp.json 42")
"""
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
        RLDecisionTree, Leaf
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
    
def attention(observation, config):
     
    """
    Uses content-wise (i.e., cosine-similarity) attention to compute features

    :observation: The observation given from the environment (an image)
    :queries: A list of queries
    :config: The configuration
    :returns: A list of (x, y) features
    """

    #preprocessing observation
    observation = observation[34:194,:,0] #cropping unnecessary parts    
    
    #f1 = find_indexes_2d(observation, 213) #other player
    f2 = find_indexes_2d(observation, 236) #the ball
    f3 = find_indexes_2d(observation, 92)  #our player
    
        
    return np.array([f2,f3]).flatten()


def save_lines_to_file(lines, file_path):
    with open(file_path, 'a') as file:  # Open file in append mode
        for line in lines:
            file.write(line + '\n')
            

def print_tree(node, depth=0):
    # Base case: If the node is None, return
    #print(type(node))
    if node is None or isinstance(node, Leaf):
        print(node)
        return
    
    # Print the current node's condition with indentation based on the depth
    indent = "    " * depth  # Increase indentation with depth
    print(f"{indent}{node}")

    # Recursive case: Traverse the left and right children
    print_tree(node.get_left(), depth + 1)
    print_tree(node.get_right(), depth + 1)
    


def evaluate(tree, config):
    """
    Evaluates the fitness of the pair composed by the given tree and the
    given queries

    :pairs: a tuple (RLDecisionTree, List of queries)
    :config: The config dictionary
    :returns: a tuple (fitness: float, trained tree: RLDecisionTree)
    """
    #print("eval:")
    #print(tree)
    
    tree = RLDecisionTree(tree, config["training"]["gamma"])
    #print_tree(tree.get_root())
    
    
    
    # Check if the tree is valid
    if tree is None:
        return ([], float("-inf"), None)
    
    env = gym.make(config["env"]["env_name"]) #render_mode = 'human'
    cum_rews = []

    #print(config["training"]["episodes"])
    # Iterate over the episodes
    for i in range(config["training"]["episodes"]):
        tree.empty_buffers()
        #env.seed(i)
        obs = env.reset(seed = i)
        done = False
        cum_rews.append(0)
        # count = 0

        while not done:
            obs = attention(obs, config)
            #print(obs)
            action = tree.get_output(obs)
            #print(action)
            obs, rew, done, _ = env.step(action)
            tree.set_reward(rew)
            
            # if(rew != 0):
            #     count = count + 1
            #     if(count == 10):
            #         done = True

            cum_rews[-1] += rew
        tree.set_reward_end_of_episode()
    
    print_tree(tree.get_root())
    print(cum_rews)

    return np.mean(cum_rews)



def produce_tree(config, log_path):
    """
    Produces a tree for the selected problem by using the Grammatical Evolution

    :config: a dictionary containing all the parameters
    :log_path: a path to the log directory
    """
    # Setup GE
    gp_config = config["gp"]

    # Build classes of the operators from the config file
    gp_config["c_factory"] = ConditionFactory()
    gp_config["l_factory"] = QLearningLeafFactory(
        config["leaves"]["params"],
        config["leaves"]["decorators"]
    )
    gp = genetic_programming.GeneticProgramming(**gp_config)

    # Retrieve the map function from utils
    map_ = utils.get_map(config["training"]["jobs"])

    # Initialize best individual
    best, best_fit, new_best = None, -float("inf"), False


    print(
        f"{'Generation' : <10} {'Min': <10} {'Mean': <10} \
        {'Max': <10} {'Std': <10} {'Invalid': <10} {'Time': <10}"
    )
        

    # Iterate over the generations
    for gen in range(config["training"]["generations"]):
        # Retrieve the current population of trees
        trees = gp.ask()

        # Retrieve the current population of queries
        #print(len(trees))

        return_values = map_(evaluate, trees, config)

        # Tell the fitnesses to the GE
        gp.tell(return_values)

        
        whole_fitnesses = return_values

        # Compute stats
        min_ = np.min(whole_fitnesses)
        mean = np.mean(whole_fitnesses)
        max_ = np.max(whole_fitnesses)
        std = np.std(whole_fitnesses)
        invalid = np.sum(np.array(whole_fitnesses) == float("-inf"))
        cur_t = time()
        
        
        line = [f"{gen: <10} {min_: <10.2f} {mean: <10.2f} \
        {max_: <10.2f} {std: <10.2f} {invalid: <10} {cur_t: <10}"]

        print(
            f"{gen: <10} {min_: <10.2f} {mean: <10.2f} \
            {max_: <10.2f} {std: <10.2f} {invalid: <10} {cur_t: <10}"
        )
        
        file_path = "output.txt"
        save_lines_to_file(line, file_path)
        
    return trees, return_values



if __name__ == "__main__":
    import json
    import utils
    import shutil
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path of the config file to use")
    parser.add_argument("seed", type=int, help="Random seed to use")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Load the config file
    config = json.load(open(args.config))
    if args.debug:
        config["training"]["jobs"] = 1

    # Set the random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Setup logging
    logdir_name = utils.get_logdir_name()
    log_path = f"logs/magent_onegenotype_oneteam/{logdir_name}"
    join = lambda x: os.path.join(log_path, x)

    os.makedirs(log_path, exist_ok=False)
    shutil.copy(args.config, join("config.json"))
    with open(join("seed.log"), "w") as f:
        f.write(str(args.seed))

    t,r = produce_tree(config, log_path)

