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
    
    
    runfile('C:/Users/Kirito/Desktop/marl_dts-master-src/marl_dts-master-src/src/experiment_launchers/pong_launcher_gp_framestacking.py',
            args="C:/Users/Kirito/Desktop/marl_dts-master-src/marl_dts-master-src/src/experiment_launchers/configs/pong_attention_gp.json 42")
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
        RLDecisionTree
from FrameStackingWrapper import FrameStackingWrapper
import cv2


def cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a, axis=1) * np.linalg.norm(b))


def fs_attention(observations, queries, config):
    
    #print(observations.shape)
    
    features = []
    
    for obs in observations:
        f = attention(obs, queries, config)
        features.extend(f)
    
    print(np.array(features).shape)
    return np.array(features)
        
    
    
def attention(observation, queries, config):
    """
    Uses content-wise (i.e., cosine-similarity) attention to compute features

    :observation: The observation given from the environment (an image)
    :queries: A list of queries
    :config: The configuration
    :returns: A list of (x, y) features
    """
    
    #preprocessing observation
    observation = observation[34:194,:,:] #cropping unnecessary parts
    #observation = np.resize(observation,(80,80,3))
    res = cv2.resize(observation, dsize=(80, 80), interpolation=cv2.INTER_CUBIC)
    
    
    h, w, _ = observation.shape
    delta = config["attention"]["query_width"]
    img_patches = []
    features = []
    
    q = queries.reshape(3,5,5,3)[0]
    print(q[:,:,0])
    from matplotlib import pyplot as plt
    plt.imshow(res, interpolation='nearest')
    plt.show()

    # FIXME: Is padding important? <29-06-21, Leonardo> #
    for j in range(0, h - delta, config["attention"]["stride"]):
        for i in range(0, w - delta, config["attention"]["stride"]):
            img_patches.append(observation[j:j+delta, i:i+delta].flatten())
    img_patches = np.array(img_patches)

    w_locations = np.floor((w - delta) / config["attention"]["stride"])
    
    print(queries.shape)
    for q in queries:
        
        attention_scores = cosine_similarity(img_patches, q)
        best = np.argmax(attention_scores)

        x = best % w_locations
        y = best // w_locations

        features.extend([x, y])
    return features


def evaluate(pairs, config):
    """
    Evaluates the fitness of the pair composed by the given tree and the
    given queries

    :pairs: a tuple (RLDecisionTree, List of queries)
    :config: The config dictionary
    :returns: a tuple (fitness: float, trained tree: RLDecisionTree)
    """
    tree, queries = pairs
    print("eval:")
    print(tree)
    
    tree = RLDecisionTree(tree, config["training"]["gamma"])
    # Check if the tree is valid
    if tree is None:
        return ([], float("-inf"), None)
    
    env = FrameStackingWrapper(gym.make(config["env"]["env_name"]), stack_size=2)
    cum_rews = []

    # Iterate over the episodes
    for i in range(config["training"]["episodes"]):
        tree.empty_buffers()
        env.seed(i)
        obs = env.reset()
        done = False
        cum_rews.append(0)

        while not done:
            obs = fs_attention(obs, queries, config)
            #print(obs)
            action = tree.get_output(obs)
            obs, rew, done, _ = env.step(action)
            tree.set_reward(rew)
            cum_rews[-1] += rew
            # env.render()
        tree.set_reward_end_of_episode()

    return np.mean(cum_rews), tree


def list_to_queries(list_, config):
    """
    Transforms a list of floats in a list of "queries", i.e. reference images

    :list_: A list of floats
    :config: Config dictionary
    :returns: A list of lists of floats
    """
    query_len = config["attention"]["query_width"] ** 2 * 3
    queries = np.array(list_).reshape((-1, query_len))
    return queries


def mix(sequence, config):
    """
    Mixes the elements of a given list

    :sequence: The list
    :config: The config dictionary
    :returns: A tuple(mixed_elements, indices of the initial list)
    """
    seq_len = config["coevolution"]["n_evaluations"]
    all_elements_contained = False
    mixed_indices = []

    while not all_elements_contained:
        all_elements_contained = True
        mixed_indices = list(np.random.permutation([*range(len(sequence))]))
        if seq_len > len(sequence):
            mixed_indices.extend(np.random.randint(0, len(sequence), seq_len - len(sequence)))

        for i in range(len(sequence)):
            if i not in mixed_indices:
                all_elements_contained = False
                break

    return ([sequence[i] for i in mixed_indices], mixed_indices)


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

    # Initialize continuous optimization algorithm
    co_config = config["continuous_opt"]
    co_config["args"]["n_params"] = (config["attention"]["query_width"] ** 2) * \
        config["attention"]["n_queries"] * 3  # RGB
    co = getattr(continuous_optimization, co_config["algorithm"])(
        **co_config["args"]
    )
    

    # Retrieve the map function from utils
    map_ = utils.get_map(config["training"]["jobs"])

    # Initialize best individual
    best, best_fit, new_best = None, -float("inf"), False

    with open(os.path.join(log_path, "log.txt"), "a") as f:
        f.write(f"Generation Min Mean Max Std Invalid Time\n")
    print(
        f"{'Generation' : <10} {'Min': <10} {'Mean': <10} \
        {'Max': <10} {'Std': <10} {'Invalid': <10} {'Time': <10}"
    )

    # Iterate over the generations
    for gen in range(config["training"]["generations"]):
        # Retrieve the current population of trees
        trees = gp.ask()

        # Retrieve the current population of queries
        qpop = co.ask()
        queries = map_(list_to_queries, qpop, config)
        
        
        #print(len(trees))
        #print(trees[0])
        #print(np.array(queries).shape)

        # Compute the fitnesses
        trees_to_evaluate, t_indices = mix(trees, config)
        queries_to_evaluate, q_indices = mix(queries, config)

        mixed_populations = [
            (a, b) for a, b in zip(trees_to_evaluate, queries_to_evaluate)
        ]
        
        
        #print(trees_to_evaluate)
        #print(queries_to_evaluate)
        

        return_values = map_(evaluate, mixed_populations, config)

        # Combine the fitnesses at the evaluation level to the individual level

        max_ = float("-inf")
        best_tree = None
        best_queries = None

        whole_fitnesses = []
        t_fitnesses_dict = {}
        q_fitnesses_dict = {}
        for index, (fitness, tree) in enumerate(return_values):
            whole_fitnesses.append(fitness)

            t_idx = t_indices[index]
            if t_idx not in t_fitnesses_dict:
                t_fitnesses_dict[t_idx] = []
            t_fitnesses_dict[t_idx].append(fitness)

            q_idx = q_indices[index]
            if q_idx not in q_fitnesses_dict:
                q_fitnesses_dict[q_idx] = []
            q_fitnesses_dict[q_idx].append(fitness)

            if fitness > max_:
                max_ = fitness
                best_tree = tree
                best_queries = queries_to_evaluate[q_idx]

        t_fitnesses_list = []
        q_fitnesses_list = []

        for i in range(len(trees)):
            t_fitnesses_list.append(np.mean(t_fitnesses_dict[i]))
        for i in range(len(qpop)):
            q_fitnesses_list.append(np.mean(q_fitnesses_dict[i]))

        # Check whether the best has to be updated
        if max_ > best_fit:
            best_fit = max_
            new_best = True

        # Tell the fitnesses to the GE
        gp.tell(t_fitnesses_list)

        # Tell the fitnesses to CO
        co.tell(q_fitnesses_list)

        # Compute stats
        min_ = np.min(whole_fitnesses)
        mean = np.mean(whole_fitnesses)
        max_ = np.max(whole_fitnesses)
        std = np.std(whole_fitnesses)
        invalid = np.sum(np.array(whole_fitnesses) == float("-inf"))
        cur_t = time()

        print(
            f"{gen: <10} {min_: <10.2f} {mean: <10.2f} \
            {max_: <10.2f} {std: <10.2f} {invalid: <10} {cur_t: <10}"
        )

        # Update the log file
        with open(os.path.join(log_path, "log.txt"), "a") as f:
            f.write(f"{gen} {min_} {mean} {max_} {std} {invalid} {cur_t}\n")
            if new_best:
                f.write(f"New best pair.\nTree: {best_tree}; \
                        Queries: {best_queries}; Fitness: {best_fit}\n")
        new_best = False
    return best


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

    produce_tree(config, log_path)
