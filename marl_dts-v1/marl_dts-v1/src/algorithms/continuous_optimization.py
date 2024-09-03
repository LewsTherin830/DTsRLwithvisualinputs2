#!/bin/python3
"""
This file contains the implementation of estimation of distribution algorithms
"""
import os
import cma
import numpy as np
#import nevergrad as ng
from .common import ContinuousOptimizationMetaClass


class ContinuousOptimizer():
    """
    A base class for continuous optimizers
    """

    def ask(self):
        """
        Returns the current population
        """
        raise NotImplementedError("This method must be implemented by the extending class")

    def tell(self, fitnesses):
        """
        Assigns the fitnesses to the members of the population

        :fitnesses: A list (with the same order as the individuals returned by tell) of floats
        """
        raise NotImplementedError("This method must be implemented by the extending class")


class EDA(ContinuousOptimizer, metaclass=ContinuousOptimizationMetaClass):
    """Ask-tell abstract-class for estimation of distribution algorithms"""

    def __init__(self, selection_size, lambda_, n_params):
        """Initializes an EDA

        :selection_size: The number of individuals used for selection
        :lambda_: The number of the samples to generate
        :n_params: The number of params to optimize

        """
        self._selection_size = selection_size
        self._lambda_ = lambda_
        self._n_params = n_params

    def ask(self):
        return self._generate()

    def tell(self, fitnesses):
        indices = sorted([*range(len(fitnesses))], key=lambda x: fitnesses[x], reverse=True)
        self._log_info("Sorted fitnesses: " + str(np.array(fitnesses)[indices]))
        return self._update(indices)

    def _generate(self):
        raise NotImplementedError("This method must be implemented by the extending class")

    def _update(self, indices):
        raise NotImplementedError("This method must be implemented by the extending class")

    def _log_info(self, fname):
        raise NotImplementedError("This method must be implemented by the extending class")


class UMDAc(EDA, metaclass=ContinuousOptimizationMetaClass):
    """Implementation of the UMDAc algorithm"""

    def __init__(self, selection_size, lambda_, n_params, mean=0, std=1, logdir=None):
        """
        Initializes an instance of the UMDAc algorithm.

        :selection_size: the number of individuals that are selected to sample the next gen
        :lambda_: the number of total individuals
        :n_params: The number of params to optimize
        :bounds: bounds to initialize mean and stdev
        :logdir: Path of the dir where the log must be saved

        """
        EDA.__init__(self, selection_size, lambda_, n_params)

        self._mean = np.zeros(self._n_params) + mean
        self._std = np.ones(self._n_params) * std
        self._pop = []
        self._logfile = os.path.join(logdir, "eda.log") if logdir is not None else None
        self._log_info("μ: {}, σ: {}".format(self._mean, self._std))

    def _generate(self):
        self._pop = np.random.normal(self._mean, self._std, (self._lambda_, self._n_params))
        self._log_info("Generated the following individuals:\n" + "\n".join(str(x).replace("\n", "") for x in self._pop))
        return self._pop

    def _update(self, indices):
        self._log_info("The best are {}".format(indices[:self._selection_size]))
        selected = self._pop[indices][:self._selection_size]
        self._mean = np.mean(selected, axis=0).flatten()
        self._std = np.std(selected, axis=0).flatten()
        for i in range(len(self._std)):
            if np.isnan(self._std[i]):
                self._std[i] = 0.01
        self._log_info("μ: {}, σ: {}".format(str(self._mean).replace("\n", ""), str(self._std).replace("\n", "")))

    def _log_info(self, msg):
        if self._logfile is not None:
            with open(self._logfile, "a") as f:
                f.write("[UMDAc] {}\n".format(msg))


class CMAES(EDA, metaclass=ContinuousOptimizationMetaClass):
    """Wrapper for CMA-ES"""

    def __init__(self, n_params, lambda_=None, mean=0, std=1):
        """
        Initializes an instance of CMA-ES

        :n_params: The number of params to optimize
        :lambda_: Population size
        """
        EDA.__init__(self, None, lambda_, n_params)

        args = {}
        if lambda_ is not None:
            args["popsize"] = lambda_
        self._cmaes = cma.CMAEvolutionStrategy([mean] * n_params, std, args)

    def _generate(self):
        self._individuals = self._cmaes.ask()
        return self._individuals.copy()

    def tell(self, fitnesses):
        self._cmaes.tell(self._individuals, -fitnesses)


# class DE(ContinuousOptimizer, metaclass=ContinuousOptimizationMetaClass):
#     """Wrapper for Differential evolution"""

#     def __init__(self, n_params, lambda_, min_, max_, **kwargs):
#         ContinuousOptimizer.__init__(self)

#         self._de = ng.optimizers.registry["DE"]
#         self._de.__init__(popsize=lambda_, **kwargs)
#         param = ng.p.Array(shape=(n_params,), lower=min_, upper=max_)
#         self._de = self._de(parametrization=param)
#         self._lambda = lambda_

#     def ask(self):
#         self._pop = []

#         for _ in range(self._lambda):
#             self._pop.append(self._de.ask())
#         return [x.value for x in self._pop]

#     def tell(self, fitnesses):
#         for x, y in zip(self._pop, fitnesses):
#             self._de.tell(x, -y)
