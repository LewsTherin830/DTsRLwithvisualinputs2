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


def evaluate(tree, config):
    """
    Evaluates the fitness of the given tree

    :tree: A tree with the common structure of the agents of the team
    :config: A dictionary with all the settings
    :returns: A float
    """
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
            agents[agent_name] = deepcopy(tree)
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

        # Iterate over all the agents
        for index, agent_name in enumerate(env.agent_iter()):
            i = index % len(agents)
            obs, rew, dones[i], _ = env.last()
            if not dones[i]:
                if agents[agent_name] is not None:
                    agents[agent_name].set_reward(rew)
                    env.step(agents[agent_name].get_output(obs.flatten()))
                else:
                    env.step(env.action_spaces[agent_name].sample())

            if index % len(agents) == 0:
                remaining_timesteps -= 1
            if remaining_timesteps < 0 or all(dones) or env.env_done:
                break

        score.append(0)
        for agent in agents.keys():
            if config["team_to_optimize"] not in agent:
                # Get the "my_team_hp" matrix
                obs = env.observe(agent)[:, :, 2]

                # Compute the center
                x = int(np.floor(obs.shape[1] / 2))
                y = int(np.floor(obs.shape[0] / 2))

                # Use damage received as score for the "team_to_optimize" team
                score[-1] += (1 - obs[y, x])

    return np.mean(score)


def produce_tree(config, log_path):
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
    map_ = utils.get_map(config["training"]["jobs"])
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
        fitnesses = map_(evaluate, trees, config)

        # Check whether the best has to be updated
        amax = np.argmax(fitnesses)
        max_ = fitnesses[amax]

        if max_ > best_fit:
            best_fit = max_
            best = pop[amax]
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

    produce_tree(config, log_path)
