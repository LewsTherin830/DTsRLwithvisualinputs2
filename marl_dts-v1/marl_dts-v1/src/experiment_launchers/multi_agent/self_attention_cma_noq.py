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


def evaluate(parameters, config):
    """
    Evaluates the fitness of the pair composed by the given tree and the
    given queries

    :pairs: a tuple (RLDecisionTree, List of queries)
    :config: The config dictionary
    :returns: a tuple (fitness: float, trained tree: RLDecisionTree)
    """
    parameters = np.array(parameters)
    tp, parameters = parameters[:13], parameters[13:]
    d = config["attention"]["d"]
    parameters = parameters.clip(-1, 1)
    w_k = parameters[:d * (3*(config["attention"]["patch_width"] ** 2) + 1)]
    w_q = parameters[d * (3*(config["attention"]["patch_width"] ** 2) + 1):]

    w_k = w_k.reshape(-1, d)
    w_q = w_q.reshape(-1, d)

    env = gym.make(config["env"]["env_name"])
    n_actions = env.action_space.n
    n_features = 4
    idx = lambda n: int(np.clip(n, 0, 0.9) * n_features)
    act = lambda n: int(np.clip(n, 0, 0.9) * n_actions)
    globals_ = {"out": None, "idx": idx, "act": act}
    cum_rews = []

    code = f"""
if f[{idx(tp[0])}] < f[{idx(tp[1])}] + {tp[2]}:
    if f[{idx(tp[3])}] < f[{idx(tp[4])}] + {tp[5]}:
        out={act(tp[6])}
    else:
        out={act(tp[7])}
else:
    if f[{idx(tp[8])}] < f[{idx(tp[9])}] + {tp[10]}:
        out={act(tp[11])}
    else:
        out={act(tp[12])}"""

    # Iterate over the episodes
    for i in range(config["training"]["episodes"]):
        env.seed(i)
        obs = env.reset()
        obs = convert_obs(obs, config)
        done = False
        cum_rews.append(0)
        steps = 0
        positives = 0

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
            globals_["f"] = features
            exec(code, globals_)
            # action = tree.get_output(features)
            action = globals_["out"]
            obs, rew, done, _ = env.step(action)
            obs = convert_obs(obs, config)
            # tree.set_reward(rew)
            cum_rews[-1] += rew
            if rew > 0:
                positives += 1
            if cum_rews[-1] <= -5 and positives == 0:
                cum_rews[-1] = -21
                break
            steps += 1
            # env.render()

        # tree.set_reward_end_of_episode()
        if config["early_stop"]["enabled"]:
            if len(cum_rews) == config["early_stop"]["episodes"]:
                if np.mean(cum_rews) <= config["early_stop"]["threshold"]:
                    break

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
    att_size = 2 * d * (3*(config["attention"]["patch_width"] ** 2) + 1)
    tree_size = 13

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

        logger.log(f"{gen} {np.min(fitnesses):.2f} {np.mean(fitnesses):.2f} {np.max(fitnesses):.2f} {np.std(fitnesses):.2f} {time() - t}")

        if np.max(fitnesses) > best:
            best = np.max(fitnesses)
            best_p = params[np.argmax(fitnesses)]
            logger.log(f"New best:\nParams: {best_p}", verbose=False)
