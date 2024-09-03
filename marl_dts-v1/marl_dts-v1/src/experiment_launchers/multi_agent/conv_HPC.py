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
import cv2
import gym
import utils
import pickle
import numpy as np
from scp import SCPClient
from time import time, sleep
from paramiko import SSHClient
from sklearn.cluster import DBSCAN
from skimage.transform import resize
from matplotlib import pyplot as plt
from algorithms import continuous_optimization, genetic_programming
from decisiontreelibrary import RLDecisionTree, ConstantLeafFactory, ConditionFactory
from experiment_launchers.multi_agent.utils import self_attention, make_patches, build_features, convert_obs


max_jobs_hpc = 30


def get_connection(config):
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(config["hpc_url"], username=config["hpc_user"])

    return ssh


def count_output_files(string):
    count = 0
    for i in range(max_jobs_hpc):
        if f"output_{i}.pkl" in string:
            count += 1
    return count


def hpc_map(evaluate, chunks, config):
    ssh = get_connection(config)

    with SCPClient(ssh.get_transport()) as scp:
        pickle.dump(evaluate, open("f.pkl", "wb"))
        scp.put(f"f.pkl", f"f.pkl")
        pickle.dump(config, open("cfg.pkl", "wb"))
        scp.put(f"cfg.pkl", f"cfg.pkl")

        for index, element in enumerate(chunks):
            pickle.dump(element, open(f"chunk_{index}.pkl", "wb"))
            scp.put(f"chunk_{index}.pkl", f"chunk_{index}.pkl")
            i, o, e = ssh.exec_command(f"./evaluate.sh f.pkl chunk_{index}.pkl cfg.pkl")
            e = e.read()
            if len(e) > 0:
                print(e)

        stdout = ""
        while count_output_files(stdout) < max_jobs_hpc:
            _, stdout, _ = ssh.exec_command(f"ls")
            stdout = str(stdout.read())
            sleep(2)

        ret_vals = []
        for index, element in enumerate(chunks):
            scp.get(f"output_{index}.pkl")
            crv = pickle.load(open(f"output_{index}.pkl", "rb"))
            ret_vals.extend(crv)

        ssh.exec_command(f"rm *.pkl")
        ssh.exec_command(f"rm jobfile.sh.*")

        return ret_vals


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

    env = gym.make(config["env"]["env_name"], **config["env"]["kwargs"])
    cum_rews = []
    im_h = config["env"]["obs_height"]
    im_w = config["env"]["obs_width"]
    offset = config["env"]["vertical_offset"]

    action_patience = config["early_stop"].get("action_patience", None)
    min_col = config["gp"]["bounds"]["float"]["min"]
    max_col = config["gp"]["bounds"]["float"]["max"]

    # Iterate over the episodes
    for i in range(config["training"]["episodes"]):
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


def init_archive(config):
    """
    Initializes the archive of samples for the sefl-attention modules
    """
    archive = []
    env = gym.make(config["env"]["env_name"], **config["env"]["kwargs"])
    i = 0
    while len(archive) < config["training"]["archive_size"]:
        env.seed(i)
        done = False
        obs = env.reset()
        im_h = config["env"]["obs_height"]
        im_w = config["env"]["obs_width"]
        offset = config["env"]["vertical_offset"]
        obs = cv2.resize(obs[offset:], (im_w, im_h))
        obs = obs / 255
        # obs = convert_obs(obs, config)
        archive.append(obs)

        while not done:
            obs, _, done, _ = env.step(env.action_space.sample())
            im_h = config["env"]["obs_height"]
            im_w = config["env"]["obs_width"]
            offset = config["env"]["vertical_offset"]
            obs = cv2.resize(obs[offset:], (im_w, im_h))
            obs = obs / 255
            # obs = convert_obs(obs, config)
            archive.append(obs)

            if len(archive) >= config["training"]["archive_size"]:
                break

        i += 1
    print(f"The archive is composed of {len(archive)} images")
    return archive


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
            predictions[-1].extend(one_hot(t.get_output(sample)[0], n_actions))

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

        max_evals = config["coevolution"]["n_evaluations_individual"]
        n_t = len(trees)
        n_p = len(params)
        n_evals = max(n_p * min(n_t, max_evals), n_t * min(n_p, max_evals))
        shuffled_trees, t_indices = mix_population(trees, n_evals, config)
        shuffled_params, p_indices = mix_population(params, n_evals, config)

        tuples = [
            [t, p] for t, p in zip(shuffled_trees, shuffled_params)
        ]

        print(f"Evaluating {len(tuples)} ({len(trees)} x {len(params)})")

        chunk_size = int(np.ceil(len(tuples) / max_jobs_hpc))
        chunks = []

        for i in range(max_jobs_hpc):
            chunks.append(tuples[i*chunk_size:(i+1)*chunk_size])

        ret_vals = hpc_map(evaluate, chunks, config)
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
