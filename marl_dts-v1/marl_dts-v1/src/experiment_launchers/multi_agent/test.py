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


def find_ball(obs):
    for y in range(obs.shape[0]):
        for x in range(obs.shape[1]):
            if np.mean(obs[y, x]) == 236:
                return [x, y]
    return [0, 0]

def find_green(obs):
    for y in range(obs.shape[0]):
        for x in range(obs.shape[1]-30, obs.shape[1]):
            if obs[y, x, 1] == 186 and obs[y, x, 0] == 92:
                return [x, y]
    return [0, 0]


scale = 2


def evaluate(config):
    """
    Evaluates the fitness of the pair composed by the given tree and the
    given queries

    :pairs: a tuple (RLDecisionTree, List of queries)
    :config: The config dictionary
    :returns: a tuple (fitness: float, trained tree: RLDecisionTree)
    """
    def policy(x):
        """
        yb = x[1]
        yp = x[3]

        if yp > yb:
            return 2
        else:
            if yp < yb - 2:
                return 3
            else:
                return 1

        return 0
        """
        xb, yb = x[:2]
        xp, yp = x[2:4]
        xpb, ypb = x[4:6]
        xpp, ypp = x[6:]
        yp += 8/scale
        ypp += 8/scale

        if xb - xpb == 0:
            m = 0
        else:
            m = (yb - ypb)/(xb - xpb)
        yfinal = yb + m * (140/scale - xb)
        # print(yfinal, yp)

        if xb - xpb == 0:
            pexp = yp
        else:
            pexp = yp + (yp - ypp)/(xb - xpb)
        diff = yfinal - pexp

        if diff <= -3:
            return 2
        elif diff >= 3:
            return 3

        return 0
        diff = x[1] - (x[3] + 4)
        if abs(diff) < 0:
            diff = 0
        vball = x[1] - x[-3]
        vcoso = x[3] - x[-1]

        # print(diff, v)
        if diff < 0 and (vball - vcoso) < 0:
            return 2
        if diff > 0 and (vball - vcoso) > 0:
            return 3
        return 0

    env = gym.make(config["env"]["env_name"])
    cum_rews = []

    # Iterate over the episodes
    for i in range(config["training"]["episodes"]):
        env.seed(i)
        obs = env.reset()
        obs = obs[35:]
        # obs = resize(obs, (160, 160))
        obs = obs[::scale, ::scale]
        done = False
        cum_rews.append(0)
        steps = 0
        features = [0, 0, 0, 0]

        while (not done) and steps < 2000:  # config["training"]["episode_len"]:
            """
            """
            features = [*find_ball(obs), *find_green(obs), *features[:4]]
            action = policy(features)
            obs, rew, done, _ = env.step(action)
            obs = obs[35:]
            # obs = resize(obs, (32, 32))
            """
            plt.imshow(obs)
            plt.show()
            """
            obs = obs[::scale, ::scale]
            cum_rews[-1] += rew
            steps += 1
            env.render()

    env.close()
    return np.mean(cum_rews)


def main(logger, config, seed, debug=False):
    from functools import partial
    from random import Random
    from time import time

    evaluate(config)
