#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    multi_agent.self_attention_gp
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This script evolves a self-attention module together with
    a DT for RL tasks

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import os
import cv2
import gym
import pickle
import numpy as np
from time import time
from sklearn.cluster import DBSCAN
from skimage.transform import resize
from matplotlib import pyplot as plt
import logging
import yaml
import json

import sys
sys.path.append(r'C:\Users\Kirito\Desktop\marl_dts-v1\marl_dts-v1')
from src import utils
sys.path.append(r'C:\Users\Kirito\Desktop\marl_dts-v1\marl_dts-v1\src')
from algorithms import continuous_optimization, genetic_programming
from decisiontrees import RLDecisionTree, ConstantLeafFactory, ConditionFactory, QLearningLeafFactory




def cosine_similarity(a, b):
    
    a = np.array(a, dtype = 'int32')
    b = np.array(b, dtype = 'int32')
    
    return np.dot(a, b)/(np.linalg.norm(a, axis=1) * np.linalg.norm(b))

def attention(observation, query):
    """
    Uses content-wise (i.e., cosine-similarity) attention to compute features

    :observation: The observation given from the environment (an image)
    :queries: A list of queries
    :config: The configuration
    :returns: A list of (x, y) features
    """
    
    h, w, _ = observation.shape
    height, width, _ = query.shape
    img_patches = []
    stride = 1
    
    query = query.flatten()
    
    # q = queries.reshape(3,5,5,3)[0]
    # print(q[:,:,0])
    # from matplotlib import pyplot as plt
    # plt.imshow(res, interpolation='nearest')
    # plt.show()
    
    for j in range(0, h - height, stride):
        for i in range(0, w - width, stride):
            img_patches.append(observation[j:j+height, i:i+width].flatten())
    img_patches = np.array(img_patches)
    
    w_locations = np.floor((w - width) / stride)
    
    attention_scores = cosine_similarity(img_patches, query)
    
    best = np.argmax(attention_scores)
    #print(attention_scores)

    x = best % w_locations
    y = best // w_locations

    return [x,y]


def evaluate(pairs, config):
    """
    Evaluates the fitness of the pair composed by the given tree and the
    given queries

    :pairs: a tuple (RLDecisionTree, List of queries)
    :config: The config dictionary
    :returns: a tuple (fitness: float, trained tree: RLDecisionTree)
    """
    tree, parameters = pairs
    tree = RLDecisionTree(tree, config["training"]["gamma"])
    #d = config["attention"]["d"]

    patches = [None, None]
    # ball = -1 * np.ones((5, 5, 5))
    # ball[1:4, 1:4] = 1
    # patches[0] = ball
    # rack = -1 * np.ones((5, 5, 5))
    # rack[2:, 2:4, 1] = 1
    # patches[1] = rack
    
    ball = np.load('ball.npy')
    rack = np.load('racket.npy')
    patches[0] = ball
    patches[1] = rack

    """
    pw = config["attention"]["patch_width"]
    patch_size = (pw ** 2) * 3
    for i in range(config["attention"]["k"]):
        w = parameters[i*patch_size:(i+1)*patch_size].reshape((pw, pw, 3))
        patches.append(w)
    """

    env = gym.make(config["env"]["env_name"]) #render_mode = "human" ) #**config["env"]["kwargs"],)
    cum_rews = []
    im_h = config["env"]["obs_height"]
    im_w = config["env"]["obs_width"]
    offset = config["env"]["vertical_offset"]

    # Iterate over the episodes
    for i in range(config["training"]["episodes"]):
        
        tree.empty_buffers()
        # env.seed(i)  # To improve speed (takes approx 14% of the time)
        obs = env.reset()
        obs = cv2.resize(obs[offset:], (im_w, im_h))
        
        
        h, w, _ = obs.shape
        done = False
        cum_rews.append(0)
        steps = 0
        positives = 0
        prev_features = None

        while (not done): #and steps < config["training"]["episode_len"]:
            
            features = []
            
            for weight in patches:
                f = attention(obs,weight)
                features.extend(f)
            features = np.array(features)

            # if prev_features is None:
            #     prev_features = features.copy()
            
            #print(features)
            #[*features, *prev_features]
            
            action = tree.get_output(np.array(features))
            #print(action)
            obs, rew, done, _ = env.step(action)
            tree.set_reward(rew)
            
            obs = cv2.resize(obs[offset:], (im_w, im_h))
            # prev_features = features.copy()
            
            if rew > 0:
                positives += 1
            
            cum_rews[-1] += rew
            steps += 1
            
            #print(cum_rews[-1])
            if cum_rews[-1] <= config["early_stop"]["threshold"] and positives == 0:
                cum_rews[-1] = config["early_stop"]["assign"]
                #print("STOPPPPP")
                break
            # env.render()

        tree.set_reward_end_of_episode()
    env.close()
    return np.mean(cum_rews), tree


def init_archive(config):
    """
    Initializes the archive of samples for the sefl-attention modules
    """
    archive = []
    env = gym.make(config["env"]["env_name"], **config["env"]["kwargs"])
    i = 0
    while len(archive) < config["training"]["archive_size"]:
        env.seed(i)
        done = False
        obs = env.reset()
        im_h = config["env"]["obs_height"]
        im_w = config["env"]["obs_width"]
        offset = config["env"]["vertical_offset"]
        obs = cv2.resize(obs[offset:], (im_w, im_h))
        obs = obs / 255
        # obs = convert_obs(obs, config)
        archive.append(obs)

        while not done:
            obs, _, done, _ = env.step(env.action_space.sample())
            im_h = config["env"]["obs_height"]
            im_w = config["env"]["obs_width"]
            offset = config["env"]["vertical_offset"]
            obs = cv2.resize(obs[offset:], (im_w, im_h))
            obs = obs / 255
            # obs = convert_obs(obs, config)
            archive.append(obs)

            if len(archive) >= config["training"]["archive_size"]:
                break

        i += 1
    print(f"The archive is composed of {len(archive)} images")
    return archive


def one_hot(action, n_actions):
    onehot = np.zeros(n_actions)
    onehot[action] = 1
    return onehot


def min_distance(predictions):
    distances = []

    predictions = np.array(predictions)

    for p in predictions:
        distances.append([])
        for o in predictions:
            distances[-1].append(np.linalg.norm(p - o))

    distances = np.array(distances)
    distances = distances.mean(axis=1)
    return np.argmin(distances)


def cluster(dbscan_params, predictions, individuals, config):
    db = DBSCAN(**dbscan_params)

    labels = db.fit_predict(predictions)
    clusters = {i: [] for i in range(max(labels) + 1)}
    for i, l in enumerate(labels):
        if l == -1:
            clusters[len(clusters)] = [i]
        else:
            clusters[l].append(i)

    representatives = []
    cluster_indices = -1 * np.ones(len(predictions), dtype=int)

    for l in clusters:
        median = min_distance([predictions[i] for i in clusters[l]])

        for i in clusters[l]:
            cluster_indices[i] = len(representatives)
        representatives.append(individuals[clusters[l][median]])

    assert cluster_indices.min() == 0
    return representatives, cluster_indices


def cluster_trees(trees, gen, config):
    distance = config["clustering"]["trees"]["decay_factor"] ** gen
    distance *= config["clustering"]["trees"]["initial_distance"]

    if "params" not in config["clustering"]["trees"]:
        config["clustering"]["trees"]["params"] = {}
    config["clustering"]["trees"]["params"]["eps"] = distance

    predictions = []
    h = config["env"]["obs_height"]
    w = config["env"]["obs_width"]
    n_samples = config["clustering"]["trees"]["n"]
    n_features = config["gp"]["bounds"]["input_index"]["max"]
    inputs = np.random.randint(0, max(h, w), (n_samples, n_features))

    for t in trees:
        predictions.append([])
        for sample in inputs:
            predictions[-1].extend(one_hot(t.get_output(sample)[0], n_features))

    return cluster(config["clustering"]["trees"]["params"], predictions, trees, config)


def cluster_params(params, archive, gen, config):
    distance = config["clustering"]["weights"]["decay_factor"] ** gen
    distance *= config["clustering"]["weights"]["initial_distance"]

    if "params" not in config["clustering"]["weights"]:
        config["clustering"]["weights"]["params"] = {}
    config["clustering"]["weights"]["params"]["eps"] = distance

    predictions = []
    #d = config["attention"]["d"]

    for parameters in params:
        patches = []
        pw = config["attention"]["patch_width"]
        patch_size = (pw ** 2) * 3
        for i in range(config["attention"]["k"]):
            w = parameters[i*patch_size:(i+1)*patch_size].reshape((pw, pw, 3))
            patches.append(w)

        predictions.append([])
        for obs in archive:
            features = []
            out = np.zeros_like(obs)
            h, w, _ = obs.shape
            for weight in patches:
                for ch in range(3):
                    out[:, :, ch] = cv2.filter2D(obs[:, :, ch], -1, weight[:, :, ch])
                tempf = np.argmax(out.sum(-1).flatten())
                x = tempf % w
                y = tempf // w
                features.extend([x, y])
            features = np.array(features)

            predictions[-1].extend(features)

    return cluster(config["clustering"]["weights"]["params"], predictions, params, config)


def copy_fitness_center(fitnesses, cluster):
    all_fitnesses = []

    for c in cluster:
        all_fitnesses.append(fitnesses[c])
    return all_fitnesses


def mix_population(sequence, n_evals, config):
    mixed_indices = []

    for i in range(n_evals // len(sequence)):
        mixed_indices.extend(np.random.permutation([*range(len(sequence))]))

    if len(mixed_indices) < n_evals:
        mixed_indices.extend(np.random.randint(0, len(sequence), (n_evals - len(mixed_indices))))

    assert len(mixed_indices) == n_evals, (len(mixed_indices), n_evals)

    return ([sequence[i] for i in mixed_indices], mixed_indices)


def main(logger, config, seed, debug=False):
    
    
    np.random.seed(seed)
    
    gp = utils.init_gp(config)

    pb = config.get("progress_bar", False)

    map_ = utils.get_map(config["training"]["jobs"], debug, pb)

    logger.info(f"Gen Min Mean Max Std Evaluations Time")
    best = -float("inf")
    best_t = None
    best_a = None
    n_trials = 5
    
    global archive
    archive = init_archive(config)
    for gen in range(config["training"]["generations"]):
        t = time()
        
        global trees
        trees = gp.ask()

        # *_clustering is in the form
        #   [pos_new_list[i0], pos_new_list[i1], ...]
        # t_clustering = [*range(len(trees))]
        if config["clustering"]["enabled"]:
            trees, t_clustering = cluster_trees(trees, gen, config)
        else:
            t_clustering = [*range(len(trees))]
        
        #print("longgggg")
        #print(len(trees))
        n_t = len(trees)

        tuples = [
            [t, None] for t in trees
        ]

        ret_vals = map_(evaluate, tuples, config)
        fitnesses = np.array([r[0] for r in ret_vals])

        fitnesses = copy_fitness_center(fitnesses, t_clustering)

        gp.tell(np.array(fitnesses))

        #logdir = logger._logdir
        #pickle.dump(gp, open(os.path.join(logdir, "gp.pkl"), "wb"))

        trees = [r[1] for r in ret_vals]
        logger.info(f"{gen} {np.min(fitnesses):.2f} {np.mean(fitnesses):.2f} {np.max(fitnesses):.2f} {np.std(fitnesses):.2f} {len(tuples)} {time() - t:.2f}")

        if np.max(fitnesses) > best:
            best = np.max(fitnesses)
            best_t = trees[np.argmax(fitnesses)]
            logger.info(f"New best:\nTree: {best_t}")

def read_config_file(config_dir, config_filename):
    # Construct the path to the configuration file
    config_path = os.path.join(config_dir, config_filename)
    
    # Check if the configuration file exists
    if os.path.exists(config_path):
        # Read the contents of the configuration file
        with open(config_path, 'r') as file:
            config_data = json.load(file)
            
        #print(config_data)
            
        return config_data
    
    else:
        print(f"Configuration file '{config_filename}' not found in directory '{config_dir}'")
        return None
            
if __name__ == "__main__":

    # Example usage
    config_dir = r'C:\Users\Kirito\Desktop\marl_dts-v1\marl_dts-v1\src\configs'  # Replace with the directory where your config file is located
    config_filename = 'pong_attention_gp.json'  # Replace with the name of your config file

    config = read_config_file(config_dir, config_filename)
    
    # Create a logger object
    logger = logging.getLogger(__name__)

    # # Set the logging level (optional)
    logger.setLevel(logging.INFO)

    # # Create a formatter
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a file handler
    file_handler = logging.FileHandler('example6.log')
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # # Example usage
    # logger.info('This is an information message')
    # logger.warning('This is a warning message')
    # logger.error('This is an error message')
    
    # Call main method with provided arguments
    main(logger, config, 42)
