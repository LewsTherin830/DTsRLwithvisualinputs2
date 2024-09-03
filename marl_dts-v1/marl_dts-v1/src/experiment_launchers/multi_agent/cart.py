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
from skimage.transform import resize
from matplotlib import pyplot as plt
from skimage.util import view_as_windows
from sklearn.feature_extraction.image import extract_patches_2d
from algorithms import continuous_optimization, genetic_programming
from decisiontreelibrary import RLDecisionTree, ConstantLeafFactory, ConditionFactory
from experiment_launchers.multi_agent.utils import build_features, convert_obs, extract_patches_2d, make_patches, self_attention


def evaluate(actions, config):
    """
    Evaluates the fitness of the pair composed by the given tree and the
    given queries

    :pairs: a tuple (RLDecisionTree, List of queries)
    :config: The config dictionary
    :returns: a tuple (fitness: float, trained tree: RLDecisionTree)
    """
    cum_rews = []
    env = gym.make(config["env"]["env_name"])
    observations = []
    itactions = iter(actions)
    action = lambda: int(np.round(next(itactions)))

    # Iterate over the episodes
    for i in range(config["training"]["episodes"]):
        env.seed(i)
        obs = env.reset()
        observations.append(obs)
        done = False
        steps = 0
        cum_rews.append(0)

        while (not done) and steps < config["training"]["episode_len"]:
            ac = action()
            ac = np.clip(ac, 0, env.action_space.n - 1)
            obs, rew, done, _ = env.step(ac)
            observations.append(obs)

            cum_rews[-1] += rew
            steps += 1
            # env.render()

    env.close()
    return np.mean(cum_rews), observations


def main(logger, config, seed, debug=False):
    from functools import partial
    from random import Random
    from time import time
    np.random.seed(seed)
    co_config = config["continuous_opt"]
    co_config["args"]["n_params"] = config["training"]["episodes"] * config["training"]["episode_len"]

    co = getattr(continuous_optimization, co_config["algorithm"])(
        **co_config["args"]
    )

    map_ = utils.get_map(config["training"]["jobs"], debug)

    logger.log(f"Gen Min Mean Max Std Time")
    best = -float("inf")
    best_p = None
    for gen in range(config["training"]["generations"]):
        t = time()
        params = co.ask()

        fitnesses = map_(evaluate, params, config)
        observations = [f[1] for f in fitnesses]
        fitnesses = np.array([f[0] for f in fitnesses])

        co.tell(fitnesses)

        if np.max(fitnesses) > best:
            best = np.max(fitnesses)
            best_p = params[np.argmax(fitnesses)]
            logger.log(f"New best:\nParams: {best_p}", verbose=False)

        logger.log(f"{gen:<4} {np.min(fitnesses):<7.2f} {np.mean(fitnesses):<7.2f} {np.max(fitnesses):<7.2f} {np.std(fitnesses):<7.2f} {time() - t:.3f} {best}")

