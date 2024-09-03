#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    src.experiment
    ~~~~~~~~~~~~~~

    This module implements the class that handles the experiments

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import os
import yaml
import random
import logging
import numpy as np
from time import time
from tasks import fitness_function
from processing_element import PEFMetaClass
from experiment_launchers.mapper import MapperMetaClass
from experiment_launchers.pipeline import PipelineFactoryMetaClass


class Experiment:
    """
    This class handles an experiment
    """

    def __init__(self, seed, logdir, **config):
        """
        Initializes an experiment

        :seed: An integer, used for seeding the random number generator
        :logdir: The dir where the results have to be logged
        :Factories: A list containing dictionaries composed of:
            - name: Name of the factory
            - kwargs: Parameters of the factory
        :Mapper: A list containing:
            - class: Name of the class of the mapper
            - kwargs: other Parameters that the mapper needs
        :PipelineFactory: A list containing:
            - class: Name of the class of the PipelineFactory
            - config: A dict containing other parameters that the factory needs
        :EvolutionaryProcess: A dictionary containing at least:
            - generations: The number of generations
        :Fitness: A dictionary containing:
            - env_name: the name of the gym environment
            - n_episodes: the number of episodes
            - seeding: whether to use the seeding for the random number generator
            - episode_length: length of the episode
            - early_stopping: dictionary for the early stopping. Currently supports:
                - threshold: threshold for termination. When the agent obtains score
                    <= threshold and no positive rewards are obtained, then the simulation
                    is terminated
                - assign: when the "threshold" criterion is met, the cumulative reward
                    associated to the simulation is set to this value
                - action_patience: terminates the simulation with the current cumulative
                    reward if an action is repeated a number of times greater or equal to
                    this parameter
                - positive_patience: terminates the simulation if a positive reward is not
                    obtained for a period of time greater or equal to this parameter
            - render: whether to render the simulations. Warning: this may increase
                training time.
            - env_params: A dict of parameters that have to be passed to the environment
        """
        self._seed = seed
        self._config = config
        pef_names = config["Factories"]
        self._pef = [
            PEFMetaClass.get(d["class_name"])(**d["kwargs"]) for d in pef_names
        ]
        self._map_class = MapperMetaClass.get(
            self._config["Mapper"]["class"]
        )
        self._pipeline_factory_class = PipelineFactoryMetaClass.get(
            self._config["PipelineFactory"]["class"]
        )
        self._logdir = logdir

        ############################## Log stuff ##############################
        # Make a backup of the configuration
        with open(os.path.join(self._logdir, "config.yaml"), "w") as outfile:
            outfile.write(yaml.dump(self._config))

        # Initialize the logger
        logname = os.path.join(self._logdir, "experiment.txt")
        best_logname = os.path.join(self._logdir, "bests.txt")

        self._logger = logging.getLogger(logname)
        self._logger.setLevel(logging.INFO)
        self._best_logger = logging.getLogger(best_logname)
        self._best_logger.setLevel(logging.INFO)
        # Setup logger for info
        fh = logging.FileHandler(logname)
        fh.setLevel(logging.INFO)
        self._logger.addHandler(fh)
        # Setup logger for bests
        fh = logging.FileHandler(best_logname)
        fh.setLevel(logging.INFO)
        self._best_logger.addHandler(fh)

    def _set_rng(self):
        """
        Sets the seed for the random number generators
        """
        random.seed(self._seed)
        np.random.seed(self._seed)
        self._logger.debug(f"Set the seed: {self._seed}")

    def _init_mapper(self):
        """
        Initializes the mapper
        :returns: An instance of Mapper
        """
        mapper = self._map_class(**self._config["Mapper"]["kwargs"])
        self._logger.debug("Mapper: initalization completed.")
        return mapper

    def _init_factory(self):
        """
        Initializes the factory
        :returns: An instance of PipelineFactory
        """
        pipeline_factory = self._pipeline_factory_class(
            self._pef,
            self._config["PipelineFactory"]["config"]
        )
        self._logger.debug("PipelineFactory: initalization completed.")
        return pipeline_factory

    def _update_best(self, gen, best_f, best, pipelines, fitnesses):
        """
        Updates the best and logs the new individual in case a new best is found

        :gen: The generation number
        :best_f: The current best fitness
        :best: The current best pipeline
        :pipelines: The population of pipelines
        :fitnesses: The list of fitnesses
        :returns: A tuple (best_f, best)
        """
        if np.max(fitnesses) > best_f:
            best_f = np.max(fitnesses)
            best = pipelines[np.argmax(fitnesses)]

            self._best_logger.info(
                f"New best at generation {gen}:\nPipeline: {best}\n"
            )
        return best_f, best

    def _pre_eval_hook(self, pipelines):
        """
        Hook method for further customization

        :pipelines: The list of pipelines
        :fitnesses: The list of fitnesses
        :returns: A tuple (pipelines, fitnesses)
        """
        return pipelines

    def _post_eval_hook(self, pipelines, fitnesses):
        """
        Hook method for further customization

        :pipelines: The list of pipelines
        :fitnesses: The list of fitnesses
        :returns: A tuple (pipelines, fitnesses)
        """
        return pipelines, fitnesses

    def run(self):
        """
        Runs the experiment.
        :returns: A tuple (fitness: float, pipeline: Pipeline)
        """
        self._set_rng()

        # Initialize the attributes
        mapper = self._init_mapper()
        pipeline_factory = self._init_factory()

        # Start the evolution
        best_fitness = -float("inf")
        best_pipeline = None

        self._logger.info(f"Gen Min Mean Max Std Evaluations Time")

        for gen in range(self._config["EvolutionaryProcess"]["generations"]):
            t = time()

            pipelines = pipeline_factory.ask_pop()

            pipelines = self._pre_eval_hook(pipelines)

            fitnesses = mapper.map(fitness_function, pipelines, self._config)

            pipelines = [f[1] for f in fitnesses]
            fitnesses = [f[0] for f in fitnesses]
            fitnesses = np.array(fitnesses)

            # For further customization
            pipelines, fitnesses = self._post_eval_hook(pipelines, fitnesses)

            pipeline_factory.tell_pop(pipelines, fitnesses)

            self._logger.info(f"{gen} {np.min(fitnesses)} {np.mean(fitnesses)} " + \
                              f"{np.max(fitnesses)} {np.std(fitnesses)} " + \
                              f"{len(fitnesses)} {time() - t}")

            best_fitness, best_pipeline = self._update_best(
                gen,
                best_fitness,
                best_pipeline,
                pipelines,
                fitnesses
            )
        return best_fitness, best_pipeline
