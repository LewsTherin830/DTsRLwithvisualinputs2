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
from decisiontreelibrary import RLDecisionTree, ConstantLeafFactory, ConditionFactory
from experiment_launchers.multi_agent.utils import self_attention, make_patches, build_features, convert_obs


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
    d = config["attention"]["d"]
    w_k = parameters[:d * (3*(config["attention"]["patch_width"] ** 2) + 1)]
    w_q = parameters[d * (3*(config["attention"]["patch_width"] ** 2) + 1):]

    w_k = w_k.reshape(-1, d)
    w_q = w_q.reshape(-1, d)

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
        positives = 0
        prev_features = None

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
            if prev_features is None:
                prev_features = features.copy()
            """
            if 0 <= steps <= 10 and steps % 2 == 0:
                toshow = obs[:]
                for x, y in features.reshape(-1, 2):
                    tox = min(x+5, toshow.shape[1] - 1)
                    toy = min(y+5, toshow.shape[0] - 1)
                    toshow[y, x:tox, :] = [1, 0, 0]
                    toshow[toy, x:tox, :] = [1, 0, 0]
                    toshow[y:toy, x, :] = [1, 0, 0]
                    toshow[y:toy, tox, :] = [1, 0, 0]
                plt.imshow(toshow)
                plt.show()
            """
            action = tree.get_output(np.array([*features, *prev_features]))
            obs, rew, done, _ = env.step(action)
            prev_features = features.copy()
            if rew > 0:
                positives += 1
            obs = convert_obs(obs, config)
            tree.set_reward(rew)
            cum_rews[-1] += rew
            steps += 1
            if cum_rews[-1] <= config["early_stop"]["threshold"] and positives == 0:
                cum_rews[-1] = config["early_stop"]["assign"]
                break
            # env.render()

        tree.set_reward_end_of_episode()

    env.close()
    return np.mean(cum_rews), tree


def main(logger, config, seed, debug=False):
    np.random.seed(seed)
    gp_config = config["gp"]

    # Build classes of the operators from the config file
    gp_config["l_factory"] = ConstantLeafFactory(
        config["leaves"]["params"]["n_actions"],
    )
    gp = genetic_programming.GeneticProgramming(**gp_config)

    # Initialize continuous optimization algorithm
    co_config = config["continuous_opt"]
    co_config["args"]["n_params"] = ((config["attention"]["patch_width"] ** 2) * 3 + 1) * config["attention"]["d"] * 2

    co = getattr(continuous_optimization, co_config["algorithm"])(
        **co_config["args"]
    )

    map_ = utils.get_map(config["training"]["jobs"], debug)

    env = gym.make(config["env"]["env_name"])
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

        aggregation_f = getattr(np, config["coevolution"]["aggregation"])
        t_fitnesses = [
            aggregation_f(fitnesses[t_indices == i]) for i in range(len(trees))
        ]

        p_fitnesses = [
            aggregation_f(fitnesses[p_indices == i]) for i in range(len(params))
        ]

        gp.tell(np.array(t_fitnesses))
        co.tell(np.array(p_fitnesses))

        logdir = logger._logdir
        pickle.dump(gp, open(os.path.join(logdir, "gp.pkl"), "wb"))
        pickle.dump(co, open(os.path.join(logdir, "co.pkl"), "wb"))

        trees = [r[1] for r in ret_vals]
        logger.log(f"{gen} {np.min(fitnesses):.2f} {np.mean(fitnesses):.2f} {np.max(fitnesses):.2f} {np.std(fitnesses):.2f} {time() - t}")

        if np.max(fitnesses) > best:
            best = np.max(fitnesses)
            best_t = shuffled_trees[np.argmax(fitnesses)]
            best_a = shuffled_params[np.argmax(fitnesses)]
            logger.log(f"New best:\nTree: {RLDecisionTree(best_t, 0)}\nAttention: {best_a}", verbose=False)
