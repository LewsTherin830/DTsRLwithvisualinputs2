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
from scipy.optimize import differential_evolution


def self_attention(w_k, w_q, image, k):
    """
    Computes the A matrix for the self-attention module
    and sums its columns as done in
    Y. Tang, D. Nguyen, e D. Ha, «Neuroevolution of Self-Interpretable Agents», arXiv:2003.08165 [cs], mar. 2020 http://arxiv.org/abs/2003.08165

    :w_k: The Key matrix
    :w_q: The Query matrix
    :image: The image to perform attention on
    :k: The number of patches to consider
    :returns: A list of top-k important patches
    """
    data = np.concatenate([image, np.ones((len(image), 1))], axis=-1)
    a = 1/np.sqrt(data.shape[1]) * np.dot(np.dot(data, w_k), np.dot(data, w_q).T)
    a = np.exp(a) / np.sum(np.exp(a), axis=1)
    assert len(a.shape) == 2

    voting = np.sum(a, axis=0)
    sorted_idx = np.argsort(voting)[:-(k+1):-1]
    """
    for i in sorted_idx:
        plt.figure()
        plt.imshow(image[i].reshape(5, 5, 3))
        plt.show()
    """
    return sorted_idx


def make_patches(image, config):
    """
    Create patches from an image

    :image: The image to decompose into patches
    :config: A dictionary with the configuration
    :returns: A list of patches
    """
    patch_size = config["attention"]["patch_width"]
    """
    all_patches = view_as_windows(
        image,
        patch_size,
        config["attention"]["stride"]
    )
    """
    patches = extract_patches_2d(image, [patch_size, patch_size])
    s = int(np.sqrt(patches.shape[0]))
    stride = config["attention"]["stride"]
    patches = patches.reshape(s, s, -1)
    patches = patches[::stride, ::stride]
    s = patches.shape[0] * patches.shape[1]
    patches = patches.reshape(s, -1)

    return patches


def build_features(indices, config):
    """
    Builds features from the indices of the patches

    :indices: The indices of the patches
    :config: The dictionary containing the config
    :returns: A 1d list of (x, y) coordinates
    """
    w = config["env"]["obs_width"]
    h = config["env"]["obs_height"]
    p_w = config["attention"]["patch_width"]
    border = p_w // 2 * 2
    stride = config["attention"]["stride"]

    n_patches_per_row = int((w - border)/ stride)
    n_patches_per_col = int((h - border)/ stride)

    coordinates = []

    for i in indices:
        x = i % n_patches_per_row
        y = i // n_patches_per_row

        coordinates.append(int(x / n_patches_per_row * (w - border)))
        coordinates.append(int(y / n_patches_per_col * (h - border)))
    return coordinates


def convert_obs(obs, config):
    obs = resize(np.array(obs[config["env"]["vertical_offset"]:], float) / 255, (config["env"]["obs_height"], config["env"]["obs_width"]), anti_aliasing=True)
    return obs


def evaluate(parameters, config):
    """
    Evaluates the fitness of the pair composed by the given tree and the
    given queries

    :pairs: a tuple (RLDecisionTree, List of queries)
    :config: The config dictionary
    :returns: a tuple (fitness: float, trained tree: RLDecisionTree)
    """
    parameters = np.array(parameters)
    tree_parameters, parameters = parameters[:10], parameters[10:]
    cf = ConditionFactory(config["conditions"]["type"])
    clf = ConstantLeafFactory(config["leaves"]["params"]["n_actions"])
    tree_parameters[:6] = tree_parameters[:6].clip(0, config["gp"]["bounds"]["input_index"]["max"] - 1)
    tree_parameters[6:] = tree_parameters[6:].clip(0, config["gp"]["bounds"]["action"]["max"] - 1)
    tree_parameters = np.array(tree_parameters, int)
    root = cf.create([tree_parameters[0], tree_parameters[1]])
    left = cf.create([tree_parameters[2], tree_parameters[3]])
    right = cf.create([tree_parameters[4], tree_parameters[5]])
    root.set_left(left)
    root.set_right(right)
    left.set_left(clf.create([tree_parameters[6]]))
    left.set_right(clf.create([tree_parameters[7]]))
    right.set_left(clf.create([tree_parameters[8]]))
    right.set_right(clf.create([tree_parameters[9]]))
    tree = RLDecisionTree(root, config["training"]["gamma"])
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

        while (not done) and steps < config["training"]["episode_len"]:
            patches = make_patches(obs, config)
            features = self_attention(
                w_k,
                w_q,
                patches,
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
            action = tree.get_output(features)
            obs, rew, done, _ = env.step(action)
            obs = convert_obs(obs, config)
            tree.set_reward(rew)
            cum_rews[-1] += rew
            steps += 1
            # env.render()

        tree.set_reward_end_of_episode()

    env.close()
    return np.mean(cum_rews)


def evaluator(candidates, args, config):
    map_ = utils.get_map(config["training"]["jobs"])

    fitnesses = map_(evaluate, candidates, config)
    print(f"{np.min(fitnesses):.2f} {np.mean(fitnesses):.2f} {np.max(fitnesses):.2f} {np.std(fitnesses):.2f} {time()}")
    return fitnesses


def generator(random, args, config):
    params = config["gp"]["bounds"]
    n_inputs = params["input_index"]["max"]
    n_actions = params["action"]["max"]

    d = config["attention"]["d"]
    att_size = 2 * d * (3*(config["attention"]["patch_width"] ** 2) + 1)
    tree_size = 10

    bound = max(n_inputs, n_actions)
    params = np.random.uniform(-bound, bound, tree_size + att_size)
    return params


def main(logger, config, seed, debug=False):
    from functools import partial
    from random import Random
    from time import time
    import inspyred
    from inspyred import ec
    from inspyred.ec import terminators
    np.random.seed(seed)
    pop_size = config["gp"]["pop_size"]

    params = config["gp"]["bounds"]
    n_inputs = params["input_index"]["max"]
    n_actions = params["action"]["max"]

    d = config["attention"]["d"]
    att_size = 2 * d * (3*(config["attention"]["patch_width"] ** 2) + 1)
    tree_size = 10

    bound = max(n_inputs, n_actions)
    f = partial(evaluator, config=config)
    g = partial(generator, config=config)
    prng = Random()
    prng.seed(time())

    ea = inspyred.ec.DEA(prng)
    ea.terminator = inspyred.ec.terminators.evaluation_termination
    final_pop = ea.evolve(generator=g,
                          evaluator=f,
                          pop_size=pop_size,
                          bounder=ec.Bounder(-bound, bound),
                          maximize=True,
                          max_evaluations=config["training"]["generations"])
    print(solution)
