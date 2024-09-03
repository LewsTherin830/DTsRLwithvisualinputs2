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
import gym
import utils
import numpy as np
from time import time
from skimage.util import view_as_windows
from sklearn.feature_extraction.image import extract_patches_2d
from algorithms import continuous_optimization, genetic_programming
from decisiontreelibrary import RLDecisionTree, QLearningLeafFactory, ConditionFactory
from experiment_launchers.multi_agent.utils import self_attention, make_patches, build_features, convert_obs


def evaluate(pair, config):
    """
    Evaluates the fitness of the pair composed by the given tree and the
    given queries

    :pairs: a tuple (RLDecisionTree, List of queries)
    :config: The config dictionary
    :returns: a tuple (fitness: float, trained tree: RLDecisionTree)
    """
    tree, parameters = pair
    tree = RLDecisionTree(tree, config["training"]["gamma"])
    d = config["attention"]["d"]
    w_k = parameters[:d * 3*(config["attention"]["patch_width"] ** 2)]
    w_q = parameters[d * 3*(config["attention"]["patch_width"] ** 2):]

    w_k = w_k.reshape(-1, d)
    w_q = w_q.reshape(-1, d)

    env = gym.make(config["env"]["env_name"])
    cum_rews = []
    scale = config["training"]["scale"]

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
            features = self_attention(
                w_k,
                w_q,
                patches,
                indexes,
                config["attention"]["k"]
            )
            features = build_features(features, config)
            features = np.array(features)
            action = tree.get_output(features)
            obs, rew, done, _ = env.step(action)
            obs = convert_obs(obs, config)
            tree.set_reward(rew)
            cum_rews[-1] += rew
            steps += 1
            # env.render()

        tree.set_reward_end_of_episode()

    env.close()
    return np.mean(cum_rews), tree


def main(logger, config, seed, debug=False):
    np.random.seed(seed)
    gp = utils.init_gp(config)

    # Initialize continuous optimization algorithm
    co_config = config["continuous_opt"]
    co_config["args"]["n_params"] = (config["attention"]["patch_width"] ** 2) * 3 * config["attention"]["d"] * 2

    co = getattr(continuous_optimization, co_config["algorithm"])(
        **co_config["args"]
    )

    map_ = utils.get_map(config["training"]["jobs"], debug)

    env = gym.make(config["env"]["env_name"])
    config["env"]["obs_width"] = env.observation_space.shape[1]
    env.close()

    logger.log(f"Gen Min Mean Max Std Time")
    best = -float("inf")
    best_t = None
    best_a = None
    n_trials = 5
    for gen in range(config["training"]["generations"]):
        t = time()
        trees = gp.ask()
        params = co.ask()

        shuffled_trees, t_indices = utils.mix_population(trees, config)
        shuffled_params, p_indices = utils.mix_population(params, config)

        tuples = [
            [t, p] for t, p in zip(shuffled_trees, shuffled_params)
        ]

        ret_vals = map_(evaluate, tuples, config)
        fitnesses = np.array([r[0] for r in ret_vals])
        t_indices = np.array(t_indices)
        p_indices = np.array(p_indices)

        t_fitnesses = [
            np.mean(fitnesses[t_indices == i]) for i in range(len(trees))
        ]

        p_fitnesses = [
            np.mean(fitnesses[p_indices == i]) for i in range(len(params))
        ]

        for trial in range(n_trials):
            # To avoid problems like
            #   Max recursion depth reached
            #   That can happen during xover/mutation
            try:
                gp.tell(np.array(t_fitnesses))
                break
            except:
                continue

        co.tell(np.array(p_fitnesses))

        trees = [r[1] for r in ret_vals]
        logger.log(f"{gen} {np.min(fitnesses)} {np.mean(fitnesses)} {np.max(fitnesses)} {np.std(fitnesses)} {time() - t}")

        if np.max(fitnesses) > best:
            best = np.max(fitnesses)
            best_t = shuffled_trees[np.argmax(fitnesses)]
            best_a = shuffled_params[np.argmax(fitnesses)]
            logger.log(f"New best:\nTree: {best_t}\nAttention: {best_a}", verbose=False)
