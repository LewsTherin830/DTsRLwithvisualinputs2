# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
    src.conv_gp_noq_alternate
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    This module implements an alternate coevolution to
    reduce the training complexity from O(n^2) to O(n)

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import os
import cv2
import gym
import utils
import pickle
import numpy as np
from time import time
from sklearn.cluster import DBSCAN
from skimage.transform import resize
from matplotlib import pyplot as plt
from algorithms import continuous_optimization, genetic_programming
from decisiontreelibrary import RLDecisionTree, ConstantLeafFactory, ConditionFactory
from experiment_launchers.multi_agent.utils import self_attention, make_patches, build_features, convert_obs
from utils import init_archive, one_hot, min_distance, cluster_trees, cluster_params, copy_fitness_center, mix_population


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

    patches = []
    pw = config["attention"]["patch_width"]
    patch_size = (pw ** 2) * 3
    for i in range(config["attention"]["k"]):
        w = parameters[i*patch_size:(i+1)*patch_size].reshape((pw, pw, 3))
        patches.append(w)

    params = config["env"]["kwargs"]
    cum_rews = []
    im_h = config["env"]["obs_height"]
    im_w = config["env"]["obs_width"]
    offset = config["env"]["vertical_offset"]

    action_patience = config["early_stop"].get("action_patience", None)
    positive_patience = config["early_stop"].get("positive_patience", None)
    min_col = config["gp"]["bounds"]["float"]["min"]
    max_col = config["gp"]["bounds"]["float"]["max"]

    # Iterate over the episodes
    for _ in range(config["training"]["episodes"]):
        env = gym.make(config["env"]["env_name"], **params)
        tree.empty_buffers()
        # env.seed(i)  # To improve speed (takes approx 14% of the time)
        obs = env.reset()
        obs = cv2.resize(obs[offset:], (im_w, im_h))
        obs = obs / 255
        # obs = convert_obs(obs, config)
        h, w, _ = obs.shape
        done = False
        cum_rews.append(0)
        steps = 0
        positives = 0
        prev_features = None
        action_ctr = 0
        last_action = None
        last_positive = 0

        while (not done) and steps < config["training"]["episode_len"]:
            features = []
            for weight in patches:
                out = np.zeros(obs.shape[:2])
                for ch in range(3):
                    out += cv2.filter2D(obs[:, :, ch], -1, weight[:, :, ch])
                tempf = np.argmax(out.flatten())
                x = tempf % im_w
                y = tempf // im_w
                features.extend([x, y])
                if config["attention"].get("use_color", False):
                    yf = min(h-1, y+pw)
                    xf = min(w-1, x+w)
                    mean_col = np.array([obs[y:yf, x:xf, c_i].mean() for c_i in range(3)])
                    normalized_col = np.array(mean_col/255 * max_col)
                    features.extend(mean_col)
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
            if last_action is not None and last_action == action:
                action_ctr += 1
            else:
                last_action = action
                action_ctr = 0

            if action_patience is not None and action_ctr >= action_patience:
                # Assume it won't get any more positive rewards
                break

            if config["env"].get("continuous_actions", False):
                action = one_hot(action, config["env"]["action_size"])
            obs, rew, done, _ = env.step(action)
            prev_features = features.copy()
            if rew > 0:
                positives += 1
                last_positive = steps
            # obs = convert_obs(obs, config)
            obs = cv2.resize(obs[offset:], (im_w, im_h))
            obs = obs / 255
            cum_rews[-1] += rew
            steps += 1
            if (steps - last_positive > positive_patience):
                # Assume it won't get any more positive rewards
                break
            if (cum_rews[-1] <= config["early_stop"]["threshold"] and positives == 0):
                cum_rews[-1] = config["early_stop"]["assign"]
                break
            # env.render()
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
    co_config["args"]["n_params"] = (config["attention"]["patch_width"] ** 2) * 3 * config["attention"]["k"]

    co = getattr(continuous_optimization, co_config["algorithm"])(
        **co_config["args"]
    )

    pb = config.get("progress_bar", False)

    map_ = utils.get_map(config["training"]["jobs"], debug, pb)

    logger.log(f"Gen Min Mean Max Std Evaluations Time")
    best = -float("inf")
    best_tree = None
    best_params = None

    for gen in range(config["training"]["generations"]):
        curtime = time()

        if gen == 0:
            trees = gp.ask()
            params = co.ask()
            shuf_trees, t_indices = mix_population(trees, len(trees) * len(params), config)
            shuf_params, p_indices = mix_population(params, len(trees) * len(params), config)
            t_indices = np.array(t_indices)
            p_indices = np.array(p_indices)
            tuples = []
            for t, p in zip(shuf_trees, shuf_params):
                tuples.append([t, p])
        else:
            if gen % 2 == 0:
                trees = gp.ask()
                tuples = [[t, best_params] for t in trees]
            else:
                params = co.ask()
                tuples = [[best_tree, p] for p in params]

        ret_vals = map_(evaluate, tuples, config)

        fitnesses = np.array([r[0] for r in ret_vals])

        if gen == 0:
            agg = getattr(np, config["coevolution"]["aggregation"])

            t_fitnesses = [
                agg(fitnesses[t_indices == i]) for i in range(len(trees))
            ]

            p_fitnesses = [
                agg(fitnesses[p_indices == i]) for i in range(len(params))
            ]

            gp.tell(np.array(t_fitnesses))
            co.tell(np.array(p_fitnesses))
        else:
            if gen % 2 == 0:
                gp.tell(fitnesses)
            else:
                co.tell(fitnesses)

        logger.log(f"{gen} {np.min(fitnesses):.2f} {np.mean(fitnesses):.2f} {np.max(fitnesses):.2f} {np.std(fitnesses):.2f} {len(tuples)} {time() - curtime:.2f}")
        if np.max(fitnesses) > best:
            best = np.max(fitnesses)
            if gen == 0:
                best_tree = trees[np.argmax(t_fitnesses)]
                best_params = params[np.argmax(p_fitnesses)]
            else:
                if gen % 2 == 0:
                    best_tree = trees[np.argmax(fitnesses)]
                else:
                    best_params = params[np.argmax(fitnesses)]
            logger.log(f"New best:\nTree: {best_tree}\nAttention: {best_params}", verbose=False)
