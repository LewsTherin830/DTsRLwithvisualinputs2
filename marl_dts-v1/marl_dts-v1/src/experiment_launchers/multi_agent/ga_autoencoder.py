#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    experiment_launchers.pz_magent_launcher
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This module allows to launch experiments for the MAgent environments
    in PettingZoo.  This module only manages the case in which we want to evolve only a team
    against random opponents.
    Not suitable for non-teamed environments such as gather, since the fitness
    function cannot give a measure of goodness in such contexts.

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import os
import sys
import gym
from time import time
import random
import numpy as np
from copy import deepcopy
sys.path.append(".")
import utils
from functools import partial
from matplotlib import pyplot as plt
from algorithms import genetic_programming
from algorithms import continuous_optimization
from algorithms.grammatical_evolution import GrammaticalEvolution, UniformMutator, \
        OnePointCrossover, TournamentSelection, NoReplacement
from decisiontreelibrary import QLearningLeafFactory, ConditionFactory, \
        RLDecisionTree


def cosine_similarity(a, b, anorm):
    return np.dot(a, b)/(anorm * np.linalg.norm(b) + 1e-6)


def attention(img_patches, w_locations, queries, cosine_similarity, config):
    """
    Uses content-wise (i.e., cosine-similarity) attention to compute features

    :observation: The observation given from the environment (an image)
    :queries: A list of queries
    :config: The configuration
    :returns: A list of (x, y) features
    """
    features = []

    for q in queries:
        attention_scores = cosine_similarity(img_patches, q.reshape(-1))
        best = np.argmax(attention_scores)

        x = best % w_locations
        y = best // w_locations

        features.extend([x, y])
    return np.array(features)


def evaluate(pairs, config):
    """
    Evaluates the fitness of the pair composed by the given tree and the
    given queries

    :pairs: a tuple (List of queries, list of images,func)
    :config: The config dictionary
    :returns: a tuple (fitness: float, trained tree: RLDecisionTree)
    """
    queries, buffer_, cos_sim, attention = pairs
    """
    plt.figure(figsize=(2,2))
    plt.imshow(queries[0])
    plt.show()
    """

    scores = []

    for obs in buffer_:
        feat = attention(queries=queries, cosine_similarity=cos_sim, config=config)
        reconstructed = np.ones_like(obs)
        reconstructed *= np.array([np.median(obs[:, :, i]) for i in range(3)], np.uint8)

        feat = np.array(feat).reshape((-1, 2))  # List of coordinates
        qh = config["attention"]["query_height"]
        qw = config["attention"]["query_width"]
        iw = obs.shape[1]
        ih = obs.shape[0]
        for idx, (x, y) in enumerate(feat):
            y = int(y)
            x = int(x)
            ystart = max(y - qh//2, 0)
            ystop = min(y + qh//2 + 1, obs.shape[0] - 1)
            xstart = max(x - qw//2, 0)
            xstop = min(x + qw//2 + 1, obs.shape[1] - 1)
            patch = queries[idx].reshape((qh, qw, 3))
            if y - qh // 2 < 0:
                patch = patch[-(y - qh//2):, :]
            if y + qh // 2 + 1 >= obs.shape[0]:
                patch = patch[:obs.shape[0] - (y + qh // 2 + 1), :]
            if x - qw // 2 < 0:
                patch = patch[:, -(x - qw // 2):]
            if x + qw // 2 + 1 >= obs.shape[1]:
                patch = patch[:,:obs.shape[1] - (x + qw // 2 + 1)]

            reconstructed[ystart:ystop, xstart:xstop] = patch
        scores.append(-np.linalg.norm(obs - reconstructed))
    """
    plt.figure()
    plt.imshow(obs)
    plt.figure()
    plt.imshow(reconstructed)
    plt.show()
    """

    return np.mean(scores)


def rectangle(img, x, y, width, height, r, g, b, angle):
    """
    Returns a rectangle
    """
    im_h, im_w, _ = img.shape
    # FIXME: Se il modulo da problemi provare a scalare <31-08-21, Leonardo> #
    max_ = 1e5
    x = int(x / max_ * im_h)
    y = int(y / max_ * im_w)
    angle = (angle % 360)/360 * 6.28
    width = int(width / max_ * im_w)
    height = int(height / max_ * im_h)

    mask = np.zeros(img.shape[:2])

    w = min(im_h, x + width)
    h = min(im_w, y + height)

    mask[y:y+h, x:x+w] = 1

    rotated_mask = np.zeros_like(mask)

    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    for j in range(im_h):
        for i in range(im_w):
            xp, yp = np.array(np.dot(rotation_matrix, [i, j]), np.uint8)
            if 0 <= xp < im_w and 0 <= yp < im_h:
                rotated_mask[yp, xp] = mask[j, i]

    rows, cols = np.where(mask == 1)
    img[rows, cols] = np.array([
        int(r/max_ * 255),
        int(g/max_ * 255),
        int(b/max_ * 255)
    ])
    return img


def circle(img, x, y, radius, r, g, b):
    """
    Returns a circle
    """
    max_ = 1e5
    im_h, im_w, _ = img.shape
    x = int(x / max_ * im_h)
    y = int(y / max_ * im_w)
    radius = int(radius / max_ * min(im_h, im_w))

    mask = np.zeros(img.shape[:2])

    for j in range(im_h):
        for i in range(im_w):
            if np.linalg.norm(np.array([x, y]) - np.array([i, j])) < radius:
                mask[j, i] = 1

    rows, cols = np.where(mask == 1)
    img[rows, cols] = np.array([
        int(r/max_ * 255),
        int(g/max_ * 255),
        int(b/max_ * 255)
    ])
    return img


def genotype_to_phenotype(genotype, config):
    """
    This function transforms the genotype into the corresponding phenotype

    :genotype: A list of integers
    :config: The dictionary containing the configuration
    :returns: A phenotype (3d np.array)
    """
    # Use the first three genes for the genotype
    background = genotype[:3]

    # Then, use all the others for the shapes
    shapes = [
        {
            "name": "rectangle",
            "func": rectangle,
            "n_params": 8
        },
        {
            "name": "circle",
            "func": circle,
            "n_params": 6
        }
    ]

    image = np.zeros(
        (
            config["attention"]["query_height"],
            config["attention"]["query_width"],
            3
        ),
        np.uint8
    )
    iterator = iter(genotype)
    while True:
        try:
            next_shape = next(iterator) % (len(shapes) + 1)

            if next_shape == len(shapes):
                break

            shape_dict = shapes[next_shape]
            args = []
            for _ in range(shape_dict["n_params"]):
                args.append(next(iterator))
            image = shape_dict["func"](image, *args)

        except StopIteration:
            break
    return image


def produce_tree(config, log_path):
    """
    Produces a tree for the selected problem by using the Grammatical Evolution

    :config: a dictionary containing all the parameters
    :log_path: a path to the log directory
    """
    # Initialize continuous optimization algorithm
    ga = GrammaticalEvolution(
        config["ge"]["pop_size"],
        UniformMutator(config["ge"]["gene_prob"], 10000),
        OnePointCrossover(),
        TournamentSelection(config["ge"]["tournament_size"]),
        NoReplacement(),
        config["ge"]["mut_pb"],
        config["ge"]["cx_pb"],
        config["ge"]["genotype_len"]
    )

    # Retrieve the map function from utils
    map_ = utils.get_map(config["training"]["jobs"])

    # Initialize best individual
    best, best_fit, new_best = None, -float("inf"), False

    with open(os.path.join(log_path, "log.txt"), "a") as f:
        f.write(f"Generation Min Mean Max Std Invalid Time\n")
    print(
        f"{'Generation' : <10} {'Min': <10} {'Mean': <10} \
        {'Max': <10} {'Std': <10} {'Invalid': <10} {'Time': <10}"
    )

    buffer_ = None
    if buffer_ is None:
        env = gym.make(config["env"]["env_name"])
        buffer_ = []

        # Iterate over the episodes
        for i in range(config["training"]["episodes"]):
            env.seed(i)
            obs = env.reset()
            done = False

            while not done:
                buffer_.append(obs[::2, ::2])

                obs, _, done, _ = env.step(env.action_space.sample())
    buffer_ = np.array(buffer_)
    indices = [np.random.choice([*range(len(buffer_))], config["training"]["n_imgs"], False)]
    buffer_ = buffer_[indices, :, :, :][0]

    observation = buffer_[0]
    h, w, _ = observation.shape
    delta_h = config["attention"]["query_height"]
    delta_w = config["attention"]["query_width"]
    w_locations = np.floor((w - delta_w) / config["attention"]["stride"])

    img_patches = []
    # FIXME: Is padding important? <29-06-21, Leonardo> #
    for j in range(0, h - delta_h, config["attention"]["stride"]):
        for i in range(0, w - delta_w, config["attention"]["stride"]):
            img_patches.append(observation[j:j+delta_h, i:i+delta_w].flatten())
    img_patches = np.array(img_patches)

    anorm = np.linalg.norm(img_patches, axis=1)
    cos_sim = partial(cosine_similarity, anorm=anorm)
    att = partial(attention, w_locations=w_locations, img_patches=img_patches)

    # Iterate over the generations
    for i in range(config["training"]["generations"]):
        # Retrieve the current population of queries
        pop = ga.ask()
        pop = [i.get_genes() for i in pop]
        queries = map_(genotype_to_phenotype, pop, config)

        indices = [*range(len(queries))]
        mixed_indices = [
            np.random.choice(
                indices,
                config["attention"]["n_queries"],
                replace=False
            ) for _ in range(len(queries))
        ]

        mixed_queries = [
            [queries[idx] for idx in mi] for mi in mixed_indices
        ]

        pairs = [
            (b, buffer_.copy(), cos_sim, att) for b in mixed_queries
        ]

        fitnesses = map_(evaluate, pairs, config)
        means = [[] for _ in range(len(pop))]
        pair_fitnesses = fitnesses.copy()

        for mi, f in zip(mixed_indices, fitnesses):
            for idx in mi:
                means[idx].append(f)
        fitnesses = [
            np.mean(m) if len(m) > 0 else 1
            for m in means
        ]
        fitnesses = np.array(fitnesses)
        fitnesses[fitnesses == 1] = 2 * np.min(fitnesses)

        # Combine the fitnesses at the evaluation level to the individual level

        max_ = float("-inf")
        best_queries = None

        q_fitnesses_dict = {}
        for index, fitness in enumerate(fitnesses):
            if fitness > max_:
                max_ = fitness
                best_queries = mixed_queries[index]

        # Check whether the best has to be updated
        if max_ > best_fit:
            best_fit = max_
            new_best = True

        # Tell the fitnesses to CO
        ga.tell(fitnesses)

        # Compute stats
        min_ = np.min(pair_fitnesses)
        mean = np.mean(pair_fitnesses)
        max_ = np.max(pair_fitnesses)
        std = np.std(pair_fitnesses)
        invalid = np.sum(np.array(pair_fitnesses) == float("-inf"))
        cur_t = time()

        print(
            f"{i: <10} {min_: <10.2f} {mean: <10.2f} \
            {max_: <10.2f} {std: <10.2f} {invalid: <10} {cur_t: <10}"
        )

        # Update the log file
        with open(os.path.join(log_path, "log.txt"), "a") as f:
            f.write(f"{i} {min_} {mean} {max_} {std} {invalid} {cur_t}\n")
            if new_best:
                f.write(f"New best.\n\
                        Queries: {best_queries}; Fitness: {best_fit}\n")
        new_best = False
    return best


if __name__ == "__main__":
    import json
    import utils
    import shutil
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path of the config file to use")
    parser.add_argument("seed", type=int, help="Random seed to use")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Load the config file
    config = json.load(open(args.config))
    if args.debug:
        config["training"]["jobs"] = 1

    # Set the random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Setup logging
    logdir_name = utils.get_logdir_name()
    log_path = f"logs/magent_onegenotype_oneteam/{logdir_name}"
    join = lambda x: os.path.join(log_path, x)

    os.makedirs(log_path, exist_ok=False)
    shutil.copy(args.config, join("config.json"))
    with open(join("seed.log"), "w") as f:
        f.write(str(args.seed))

    produce_tree(config, log_path)
