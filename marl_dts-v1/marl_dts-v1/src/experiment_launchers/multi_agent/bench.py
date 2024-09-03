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
import gym
import utils
import pickle
import numpy as np
from time import time
from skimage.transform import resize
from matplotlib import pyplot as plt
from skimage.util import view_as_windows
from sklearn.feature_extraction.image import extract_patches_2d
from algorithms import continuous_optimization, genetic_programming
from decisiontreelibrary import FastDecisionTree, ConstantLeafFactory, ConditionFactory
from experiment_launchers.multi_agent.utils import build_features, convert_obs, extract_patches_2d, make_patches, self_attention


def evaluate(parameters, config):
    """
    Evaluates the fitness of the pair composed by the given tree and the
    given queries

    :pairs: a tuple (RLDecisionTree, List of queries)
    :config: The config dictionary
    :returns: a tuple (fitness: float, trained tree: RLDecisionTree)
    """
    parameters = np.array(parameters)
    tree_parameters, parameters = parameters[:4], parameters[4:]
    cf = ConditionFactory(config["conditions"]["type"])
    clf = ConstantLeafFactory(config["leaves"]["params"]["n_actions"])
    tree_parameters[:2] = np.clip(tree_parameters[:2], 0, 1) * config["gp"]["bounds"]["input_index"]["max"] - 1
    tree_parameters[2:] = np.where(tree_parameters[2:] > 0, 2, 3)
    root = cf.create([int(tree_parameters[0]), int(tree_parameters[1])])
    # left = cf.create([int(tree_parameters[2]), int(tree_parameters[3])])
    # right = cf.create([int(tree_parameters[4]), int(tree_parameters[5])])
    # root.set_left(left)
    # root.set_right(right)
    root.set_left(clf.create([int(tree_parameters[2])]))
    root.set_right(clf.create([int(tree_parameters[3])]))
    # left.set_left(clf.create([int(tree_parameters[6])]))
    # left.set_right(clf.create([int(tree_parameters[7])]))
    # right.set_left(clf.create([int(tree_parameters[8])]))
    # right.set_right(clf.create([int(tree_parameters[9])]))
    tree = FastDecisionTree(root)
    d = config["attention"]["d"]
    att1_parameters, att2_parameters = parameters[:len(parameters)//2], parameters[len(parameters)//2:]
    w_k1 = att1_parameters[:d * (3*(config["attention"]["patch_width"] ** 2) + 1)]
    w_q1 = att1_parameters[d * (3*(config["attention"]["patch_width"] ** 2) + 1):]
    w_k2 = att2_parameters[:d * (3*(config["attention"]["patch_width"] ** 2) + 1)]
    w_q2 = att2_parameters[d * (3*(config["attention"]["patch_width"] ** 2) + 1):]

    w_k1 = w_k1.reshape(-1, d)
    w_q1 = w_q1.reshape(-1, d)
    w_k2 = w_k2.reshape(-1, d)
    w_q2 = w_q2.reshape(-1, d)

    env = gym.make(config["env"]["env_name"])
    cum_rews = []

    # Iterate over the episodes
    for i in range(config["training"]["episodes"]):
        tree.empty_buffers()
        env.seed(i)
        obs = env.reset()
        obs = convert_obs(obs, config)
        done = False
        cum_rews.append(0)
        steps = 0

        while (not done) and steps < config["training"]["episode_len"]:
            patches, indexes = make_patches(obs, config)
            features1 = self_attention(
                w_k1,
                w_q1,
                patches,
                indexes,
                1
            )

            features2 = self_attention(
                w_k2,
                w_q2,
                patches,
                indexes,
                1
            )

            features = np.array([features1, features2])

            features = build_features(features, config)
            features = np.array(features)

            # action = 2 if features[1] < features[3] else 3
            action = tree.get_output(features)
            obs, rew, done, _ = env.step(action)
            obs = convert_obs(obs, config)
            cum_rews[-1] += rew
            steps += 1
            # env.render()

    env.close()
    return np.mean(cum_rews)


def main(logger, config, seed, debug=False):
    from functools import partial
    from random import Random
    from time import time
    np.random.seed(seed)
    pop_size = config["gp"]["pop_size"]

    params = config["gp"]["bounds"]
    n_inputs = params["input_index"]["max"]
    n_actions = params["action"]["max"]

    d = config["attention"]["d"]
    att_size = 2 * 2 * d * (3*(config["attention"]["patch_width"] ** 2) + 1)
    tree_size = 4

    bound = max(n_inputs, n_actions)

    # Initialize continuous optimization algorithm
    co_config = config["continuous_opt"]
    co_config["args"]["n_params"] = att_size + tree_size

    co = getattr(continuous_optimization, co_config["algorithm"])(
        **co_config["args"]
    )

    map_ = utils.get_map(config["training"]["jobs"], debug)

    env = gym.make(config["env"]["env_name"])
    env.close()

    logger.log(f"Gen Min Mean Max Std Time")
    best = -float("inf")
    best_p = None
    for gen in range(config["training"]["generations"]):
        t = time()
        params = co.ask()

        fitnesses = map_(evaluate, params, config)
        fitnesses = np.array(fitnesses)

        co.tell(fitnesses)
        pickle.dump(co, open(os.path.join(logger._logdir, "co.pkl"), "wb"))

        logger.log(f"{gen} {np.min(fitnesses):.2f} {np.mean(fitnesses):.2f} {np.max(fitnesses):.2f} {np.std(fitnesses):.2f} {time() - t}")

        if np.max(fitnesses) > best:
            best = np.max(fitnesses)
            best_p = params[np.argmax(fitnesses)]
            logger.log(f"New best:\nParams: {best_p}", verbose=False)
