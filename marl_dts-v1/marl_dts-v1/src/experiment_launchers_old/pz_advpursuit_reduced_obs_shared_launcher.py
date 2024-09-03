#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    experiment_launchers.pz_magent_launcher
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This module allows to launch experiments for the MAgent environments
    in PettingZoo.
    This module only manages the case in which we want to evolve only a team
    against random opponents.
    Not suitable for non-teamed environments such as gather, since the fitness
    function cannot give a measure of goodness in such contexts.

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import os
import sys
sys.path.append(".")
import time
import utils
import random
import pettingzoo
import numpy as np
from copy import deepcopy
from algorithms import grammatical_evolution
from decisiontreelibrary import QLearningLeafFactory, ConditionFactory, \
        RLDecisionTree
from pettingzoo.magent import adversarial_pursuit_v2, battle_v2, \
        battlefield_v2, combined_arms_v3, tiger_deer_v3


def compute_features(obs):
    """
    This function computes the compressed features given the raw
    features from the simulator.

    :obs: A set of features for the agent
    :returns: A list of features
    """

    map_w = obs.shape[1]
    map_h = obs.shape[0]

    new_features = []

    obstacles = obs[:, :, 0].flatten()
    new_features.extend(obstacles)

    # Compute the number of teammates to the left, right, up and down
    #   ignoring the dead ones (hp = 0)
    nondead = (obs[:, :, 1] * obs[:, :, 2]) > 0
    # Predators have size 2x2
    const = 1 * (config["team_to_optimize"] == "predator")
    n_teammates_left = np.sum(nondead[:, :map_w // 2 - const])
    n_teammates_up = np.sum(nondead[:map_h // 2 - const, :])
    n_teammates_right = np.sum(nondead[:, const + 1 + map_w // 2:])
    n_teammates_down = np.sum(nondead[const + 1 + map_h // 2:, :])

    new_features.extend([
        n_teammates_left,
        n_teammates_up,
        n_teammates_right,
        n_teammates_down
    ])

    new_features.append(obs[map_h // 2, map_w // 2, 2])

    # Compute the number of preys to the left, right, up and down
    #   ignoring the dead ones (hp = 0)
    nondead = (obs[:, :, 3] * obs[:, :, 4]) > 0
    n_preys_left = np.sum(nondead[:, :map_w // 2])
    n_preys_up = np.sum(nondead[:map_h // 2, :])
    n_preys_right = np.sum(nondead[:, 1+map_w // 2:])
    n_preys_down = np.sum(nondead[1+map_h // 2:, :])

    new_features.extend([
        n_preys_left,
        n_preys_up,
        n_preys_right,
        n_preys_down
    ])

    prey_presence = []

    for y in [-1, 0, 1]:
        for x in [-1, 0, 1]:
            prey_presence.append(
                nondead[map_h // 2 + y, map_w // 2 + x]
            )

    new_features.extend(prey_presence)
    return np.array(new_features)


def evaluate(tree, config):
    """
    Evaluates the fitness of the given tree

    :tree: A tree with the common structure of the agents of the team
    :config: A dictionary with all the settings
    :returns: A float
    """
    # Check whether the phenotype is valid
    if tree is None:
        return -10**3, None

    # Re-import the environments here to avoid problems with parallelization
    from pettingzoo.magent import adversarial_pursuit_v2, battle_v2, \
        battlefield_v2, combined_arms_v3, tiger_deer_v3
    agents = {}
    env = getattr(pettingzoo.magent, config["env"]["env_name"])
    env = env.env(**config["env"]["env_kwargs"])
    env.reset()
    tot_agents = 0

    for agent_name in env.agents:
        agent_class = "_".join(agent_name.split("_")[:-1])
        if agent_class == config["team_to_optimize"]:
            agents[agent_name] = tree.copy()
            tot_agents += 1
        else:
            agents[agent_name] = None

    score = []

    # Start the training
    for i in range(config["training"]["episodes"]):
        # Seed the environment
        env.seed(i)
        env.reset()

        # Set max timesteps
        remaining_timesteps = config["training"]["episode_length"]
        dones = [False for _ in range(len(agents))]

        score.append(0)
        tree.empty_buffers()
        # Iterate over all the agents
        for index, agent_name in enumerate(env.agent_iter()):
            i = index % len(agents)
            obs, rew, dones[i], _ = env.last()
            if not dones[i]:
                if agents[agent_name] is not None:
                    agents[agent_name].set_reward(rew)
                    action = agents[agent_name].get_output(
                        compute_features(obs)
                    )
                    env.step(action)
                else:
                    if rew == -1:
                        # Prey tagged (In adv pursuit hp do not decrease)
                        score[-1] += 1
                    env.step(env.action_spaces[agent_name].sample())

            if index % len(agents) == 0:
                # env.render()
                remaining_timesteps -= 1
            if remaining_timesteps < 0 or all(dones) or env.env_done:
                # print(score[-1])
                break

    # env.close()
    # print(tree)
    # print()
    return np.mean(score), tree


def produce_tree(config, log_path, debug=False):
    """
    Produces a tree for the selected problem by using the Grammatical Evolution

    :config: a dictionary containing all the parameters
    :log_path: a path to the log directory
    """
    # Setup GE
    ge_config = config["ge"]

    # Build classes of the operators from the config file
    for op in ["mutation", "crossover", "selection", "replacement"]:
        ge_config[op] = getattr(
            grammatical_evolution, ge_config[op]["type"]
        )(**ge_config[op]["params"])

    ge = grammatical_evolution.GrammaticalEvolution(**ge_config)
    # Retrieve the map function from utils
    map_ = utils.get_map(config["training"]["jobs"], debug)
    # Initialize best individual
    best, best_fit, new_best = None, -float("inf"), False

    with open(os.path.join(log_path, "log.txt"), "a") as f:
        f.write(f"Generation Min Mean Max Std\n")
    print(f"{'Generation' : <10} {'Min': <10} {'Mean': <10} \
{'Max': <10} {'Std': <10}")
    # Iterate over the generations
    for i in range(config["training"]["generations"]):
        # Retrieve the current population
        pop = ge.ask()
        # Convert the genotypes in phenotypes
        trees = map_(utils.genotype2phenotype, pop, config)
        # Compute the fitnesses
        # We need to return the trees in order to retrieve the
        #   correct values for the leaves when using the
        #   parallelization
        return_values = map_(evaluate, trees, config)
        fitnesses = [r[0] for r in return_values]
        trees = [r[1] for r in return_values]

        # Check whether the best has to be updated
        amax = np.argmax(fitnesses)
        max_ = fitnesses[amax]

        if max_ > best_fit:
            best_fit = max_
            best = trees[amax]
            new_best = True

        # Tell the fitnesses to the GE
        ge.tell(fitnesses)

        # Compute stats
        min_ = np.min(fitnesses)
        mean = np.mean(fitnesses)
        max_ = np.max(fitnesses)
        std = np.std(fitnesses)

        print(f"{i: <10} {min_: <10.2f} {mean: <10.2f} \
{max_: <10.2f} {std: <10.2f}")

        # Update the log file
        with open(os.path.join(log_path, "log.txt"), "a") as f:
            f.write(f"{i} {min_} {mean} {max_} {std}\n")
            if new_best:
                f.write(f"New best: {best}; Fitness: {best_fit}\n")
                with open(join("best_tree.mermaid"), "w") as f:
                    f.write(str(best))
        new_best = False
    return best


if __name__ == "__main__":
    import json
    import utils
    import shutil
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path of the config file to use")
    parser.add_argument("--debug", action="store_true", help="Debug flag")
    parser.add_argument("seed", type=int, help="Random seed to use")
    args = parser.parse_args()

    # Load the config file
    config = json.load(open(args.config))

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

    best = produce_tree(config, log_path, args.debug)
