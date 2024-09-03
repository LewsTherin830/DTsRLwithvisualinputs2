# -*- coding: utf-8 -*-
"""
    multi_agent.self_attention_gp_conv_curriculum
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This script evolves a self-attention module together with
    a DT for RL tasks

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
    f0 = config["env"]["frameskip"]["min"]
    frameskips = np.arange(
        f0 + 1,
        config["env"]["frameskip"]["max"],
        config["env"]["frameskip"]["step"],
        dtype=int
    )
    frameskips = [[f0, f] for f in frameskips]
    cum_rews = []
    im_h = config["env"]["obs_height"]
    im_w = config["env"]["obs_width"]
    offset = config["env"]["vertical_offset"]

    action_patience = config["early_stop"].get("action_patience", None)
    min_col = config["gp"]["bounds"]["float"]["min"]
    max_col = config["gp"]["bounds"]["float"]["max"]

    # Iterate over the episodes
    for fs in frameskips:
        for i in range(config["training"]["episodes"]):
            params["frameskip"] = fs
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
                if (steps - last_positive > config["early_stop"]["positive_patience"]):
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
    best_t = None
    best_a = None
    n_trials = 5
    archive = init_archive(config)
    for gen in range(config["training"]["generations"]):
        t = time()
        trees = gp.ask()
        params = co.ask()

        # *_clustering is in the form
        #   [pos_new_list[i0], pos_new_list[i1], ...]
        # t_clustering = [*range(len(trees))]
        if config["clustering"]["enabled"]:
            trees, t_clustering = cluster_trees(trees, gen, config)
            params, p_clustering = cluster_params(params, archive, gen, config)
        else:
            t_clustering = [*range(len(trees))]
            p_clustering = [*range(len(params))]

        print(len(trees), len(params))
        max_evals = config["coevolution"]["n_evaluations_individual"]
        n_t = len(trees)
        n_p = len(params)
        n_evals = max(n_p * min(n_t, max_evals), n_t * min(n_p, max_evals))
        shuffled_trees, t_indices = mix_population(trees, n_evals, config)
        shuffled_params, p_indices = mix_population(params, n_evals, config)

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

        t_fitnesses = copy_fitness_center(t_fitnesses, t_clustering)
        p_fitnesses = copy_fitness_center(p_fitnesses, p_clustering)

        gp.tell(np.array(t_fitnesses))
        co.tell(np.array(p_fitnesses))

        logdir = logger._logdir
        pickle.dump(gp, open(os.path.join(logdir, "gp.pkl"), "wb"))
        pickle.dump(co, open(os.path.join(logdir, "co.pkl"), "wb"))

        trees = [r[1] for r in ret_vals]
        logger.log(f"{gen} {np.min(fitnesses):.2f} {np.mean(fitnesses):.2f} {np.max(fitnesses):.2f} {np.std(fitnesses):.2f} {len(tuples)} {time() - t:.2f}")

        if np.max(fitnesses) > best:
            best = np.max(fitnesses)
            best_t = shuffled_trees[np.argmax(fitnesses)]
            best_a = shuffled_params[np.argmax(fitnesses)]
            logger.log(f"New best:\nTree: {RLDecisionTree(best_t, 0)}\nAttention: {best_a}", verbose=False)
