#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    src.utils
    ~~~~~~~~~

    This module implements utilities

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import os
import gym
import cv2
import string
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.cluster import DBSCAN
from joblib import Parallel, delayed
from algorithms import genetic_programming
from decisiontrees import RLDecisionTree
from decisiontrees import ConditionFactory, QLearningLeafFactory


def get_logdir_name():
    """
    Returns a name for the dir
    :returns: a name in the format 'dd-mm-yyyy_:mm:ss_<random string>'
    """
    time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    rand_str = "".join(np.random.choice([*string.ascii_lowercase], 8))
    return f"{time}_{rand_str}"


def get_map(n_jobs, debug=False, progressbar=False):
    """
    Returns a function pointer that implements a parallel map function

    :n_jobs: The number of workers to spawn
    :debug: a flag that disables multiprocessing
    :progressbar: a flag that enables the progress bar
    :returns: A function pointer

    """
    if debug:
        def fcn(function, iterable, config):
            ret_vals = []
            for i in iterable:
                ret_vals.append(function(i, config))
            return ret_vals
    else:
        def fcn(function, iterable, config):
            if not progressbar:
                with Parallel(n_jobs) as p:
                    return p(delayed(function)(elem, config) for elem in iterable)
            else:
                with Parallel(n_jobs) as p:
                    return p(delayed(function)(elem, config) for elem in tqdm(iterable))
    return fcn


class CircularList(list):

    """
    A list that, when indexed outside its bounds (index i), returns the
    element in position i % len(self)
    """
    def __init__(self, iterable):
        """
        Initializes the list.
        If iterable is a dict, then an arange object is created as the list
        """
        if isinstance(iterable, dict):
            list.__init__(self, np.arange(**iterable))
        else:
            list.__init__(self, iterable)

    def __getitem__(self, index):
        return super().__getitem__(index % len(self))


class Grammar(dict):
    """
    Implements a Grammar, simply a dictionary of circular lists, i.e.
    lists that return an element even if the index required is outside their
    bounds
    """

    def __init__(self, dictionary):
        """
        Initializes the grammar

        :dictionary: A dictionary containing the grammar
        """
        circular_dict = {}
        for k, v in dictionary.items():
            circular_dict[k] = CircularList(v)
        dict.__init__(self, circular_dict)


def genotype2phenotype(individual, config):
    """
    Converts a genotype in a phenotype

    :individual: An Individual (algorithms.grammatical_evolution)
    :config: A dictionary
    :returns: An instance of RLDecisionTree
    """
    genotype = individual.get_genes()
    gene = iter(genotype)
    grammar = Grammar(config["grammar"])
    cfactory = ConditionFactory(config["conditions"]["type"])
    lfactory = QLearningLeafFactory(
        config["leaves"]["params"],
        config["leaves"]["decorators"]
    )
    lambda_ = config["training"].get("lambda", 0)

    if grammar["root"][next(gene)] == "condition":
        params = cfactory.get_trainable_parameters()
        root = cfactory.create(
            [grammar[p][next(gene)] for p in params]
        )
    else:
        root = lfactory.create()
        return RLDecisionTree(root, config["training"]["gamma"], lambda_)

    fringe = [root]

    try:
        while len(fringe) > 0:
            node = fringe.pop(0)

            for i, n in enumerate(["left", "right"]):
                if grammar["root"][next(gene)] == "condition":
                    params = cfactory.get_trainable_parameters()
                    newnode = cfactory.create(
                        [grammar[p][next(gene)] for p in params]
                    )
                    getattr(node, f"set_{n}")(newnode)
                    fringe.insert(i, newnode)
                else:
                    leaf = lfactory.create()
                    getattr(node, f"set_{n}")(leaf)
    except StopIteration:
        return None
    return RLDecisionTree(root, config["training"]["gamma"], lambda_)


def genotype2str(genotype, config):
    """
    Transforms a genotype in a string given the grammar in config

    :individual: An Individual algorithms.grammatical_evolution
    :config: A dictionary with the "grammar" key
    :returns: A string

    """
    pass


def init_gp(config):
    """
    Initializes GP for RL tasks with Q learning leaves
    """
    gp_config = config["gp"]

    # Build classes of the operators from the config file
    gp_config["c_factory"] = ConditionFactory()
    gp_config["l_factory"] = QLearningLeafFactory(
    leaf_params=config["leaves"]["params"],
    decorators=config["leaves"]["decorators"]
)

    
    gp = genetic_programming.GeneticProgramming(**gp_config)
    return gp


def one_hot(action, n_actions):
    onehot = np.zeros(n_actions)
    onehot[action] = 1
    return onehot


def min_distance(predictions):
    distances = []

    predictions = np.array(predictions)

    for p in predictions:
        distances.append([])
        for o in predictions:
            distances[-1].append(np.linalg.norm(p - o))

    distances = np.array(distances)
    distances = distances.mean(axis=1)
    return np.argmin(distances)


def cluster(dbscan_params, predictions, individuals, config):
    db = DBSCAN(**dbscan_params)

    labels = db.fit_predict(predictions)
    clusters = {i: [] for i in range(max(labels) + 1)}
    for i, l in enumerate(labels):
        if l == -1:
            clusters[len(clusters)] = [i]
        else:
            clusters[l].append(i)

    representatives = []
    cluster_indices = -1 * np.ones(len(predictions), dtype=int)

    for l in clusters:
        median = min_distance([predictions[i] for i in clusters[l]])

        for i in clusters[l]:
            cluster_indices[i] = len(representatives)
        representatives.append(individuals[clusters[l][median]])

    assert cluster_indices.min() == 0
    return representatives, cluster_indices


def cluster_trees(trees, gen, config):
    distance = config["clustering"]["trees"]["decay_factor"] ** gen
    distance *= config["clustering"]["trees"]["initial_distance"]

    if "params" not in config["clustering"]["trees"]:
        config["clustering"]["trees"]["params"] = {}
    config["clustering"]["trees"]["params"]["eps"] = distance

    predictions = []
    h = config["env"]["obs_height"]
    w = config["env"]["obs_width"]
    n_samples = config["clustering"]["trees"]["n"]
    n_features = config["gp"]["bounds"]["input_index"]["max"]
    if config["attention"].get("use_color", False):
        inputs = np.random.randint(0, max(h, w), (n_samples, n_features))
    else:
        inputs = np.random.randint(0, max(h, w, 255), (n_samples, n_features))
    action_dict = config["gp"]["bounds"]["action"]
    n_actions = action_dict["max"] - action_dict["min"]

    for t in trees:
        predictions.append([])
        for sample in inputs:
            predictions[-1].extend(one_hot(t.get_output(sample)[0] - action_dict["min"], n_actions))

    return cluster(config["clustering"]["trees"]["params"], predictions, trees, config)


def cluster_params(params, archive, gen, config):
    distance = config["clustering"]["weights"]["decay_factor"] ** gen
    distance *= config["clustering"]["weights"]["initial_distance"]

    if "params" not in config["clustering"]["weights"]:
        config["clustering"]["weights"]["params"] = {}
    config["clustering"]["weights"]["params"]["eps"] = distance

    predictions = []
    d = config["attention"]["d"]

    for parameters in params:
        patches = []
        pw = config["attention"]["patch_width"]
        patch_size = (pw ** 2) * 3
        for i in range(config["attention"]["k"]):
            w = parameters[i*patch_size:(i+1)*patch_size].reshape((pw, pw, 3))
            patches.append(w)

        predictions.append([])
        for obs in archive:
            features = []
            out = np.zeros_like(obs)
            h, w, _ = obs.shape
            for weight in patches:
                for ch in range(3):
                    out[:, :, ch] = cv2.filter2D(obs[:, :, ch], -1, weight[:, :, ch])
                tempf = np.argmax(out.sum(-1).flatten())
                x = tempf % w
                y = tempf // w
                features.extend([x, y])
            features = np.array(features)

            predictions[-1].extend(features)

    return cluster(config["clustering"]["weights"]["params"], predictions, params, config)


def copy_fitness_center(fitnesses, cluster):
    all_fitnesses = []

    for c in cluster:
        all_fitnesses.append(fitnesses[c])
    return all_fitnesses


def mix_population(sequence, n_evals, config):
    mixed_indices = []

    for i in range(n_evals // len(sequence)):
        mixed_indices.extend(np.random.permutation([*range(len(sequence))]))

    if len(mixed_indices) < n_evals:
        mixed_indices.extend(np.random.randint(0, len(sequence), (n_evals - len(mixed_indices))))

    assert len(mixed_indices) == n_evals, (len(mixed_indices), n_evals)

    return ([sequence[i] for i in mixed_indices], mixed_indices)


def preprocessing(obs, resize, im_w, im_h, normalize, norm_min, norm_max, cropping=None):
    """
    Preprocesses the observation

    :obs: The observation
    :resize: Flag, resizes the observation (for images) if True
    :im_w: The new width of the image
    :im_h: The new height of the image
    :normalize: Flag, normalizes the observation if True
    :norm_min: The minimum for the normalization
    :norm_max: The maximum for the normalization
    :cropping: A list of the number of pixel that have to be
    cropped at [top, bottom, left, right]. None means no cropping. Default: None
    :returns: An observation
    """
    if cropping is None:
        t, b, l, r = [0] * 4
    else:
        t, b, l, r = cropping

    h, w, _ = obs.shape

    if resize:
        obs = cv2.resize(obs[t:h-b, l:w-r], (im_w, im_h))

    if normalize:
        obs = (obs - norm_min) / (norm_max - norm_min)

    return obs


def fitness_function(pipeline, config):
    """
    Evaluates the fitness of the pipeline

    :pipeline: The pipeline to evaluate
    :config: The dictionary containing the parameters
    :returns: A tuple (fitness, trained pipeline)
    """
    fit_dict = config["Fitness"]
    seeding = fit_dict.get("seeding", False)
    episode_length = fit_dict.get("episode_length", None)
    early_stop = fit_dict.get("early_stopping", {})
    threshold = early_stop.get("threshold", None)
    threshold_assign = early_stop.get("assign", None)
    action_patience = early_stop.get("action_patience", None)
    positive_patience = early_stop.get("positive_patience", None)
    render = fit_dict.get("render", False)
    env_params = fit_dict.get("env_params", [])

    if isinstance(env_params, dict):
        # Check when there is only one set of parameters
        env_params = [env_params]

    cum_rews = []

    for cur_params in env_params:
        env = gym.make(fit_dict["env_name"], **cur_params)
        # Iterate over the episodes
        for i in range(fit_dict["n_episodes"]):
            pipeline.new_episode()
            if seeding:
                env.seed(i)  # Do not use it to improve speed (takes approx 14% of the time)
            obs = env.reset()

            done = False
            cum_rews.append(0)

            steps = 0
            positives = 0
            last_positive = 0
            action_ctr = 0
            last_action = None

            while (not done) and (episode_length is None or steps < episode_length):
                action = pipeline.get_output(obs)
                if not isinstance(action, int):
                    action = np.argmax(action)

                if last_action is not None and last_action == action:
                    action_ctr += 1
                else:
                    last_action = action
                    action_ctr = 0

                if action_patience is not None and action_ctr >= action_patience:
                    # Assume it won't get any more positive rewards
                    break

                obs, rew, done, _ = env.step(action)

                if rew > 0:
                    positives += 1
                    last_positive = steps

                cum_rews[-1] += rew
                pipeline.set_reward(rew)
                steps += 1

                if positive_patience is not None and \
                   (steps - last_positive > positive_patience):
                    # Assume it won't get any more positive rewards
                    break
                if threshold is not None and \
                   (cum_rews[-1] <= threshold and positives == 0):
                    cum_rews[-1] = threshold_assign
                    break
                if render:
                    env.render()
        env.close()
    return np.mean(cum_rews), pipeline


def get_output_pipeline(data, config):
    pipeline, archive = data
    outputs = []
    for x in archive:
        outputs.append(pipeline.get_output(x))
    outputs = np.array(outputs)
    return outputs.reshape(len(data), -1)
