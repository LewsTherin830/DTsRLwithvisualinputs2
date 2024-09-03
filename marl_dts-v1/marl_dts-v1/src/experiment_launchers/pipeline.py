#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    experiment_launchers.pipeline
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This module implements the pipeline entity and their factories

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import numpy as np
from abc import abstractmethod


class PipelineFactoryMetaClass(type):
    _registry = {}

    def __init__(cls, clsname, bases, methods):
        super().__init__(clsname, bases, methods)
        PipelineFactoryMetaClass._registry[cls.__name__] = cls

    @staticmethod
    def get(class_name):
        """
        Retrieves the class associated to the string

        :class_name: The name of the class
        :returns: A class
        """
        return PipelineFactoryMetaClass._registry[class_name]


class Pipeline:

    """
    This class implements a pipeline, i.e., a collection of processing
    elements that are queried sequentially and each module's input corresponds
    to the output of the previous element.
    The first element takes in input the input coming from the environment.
    """

    def __init__(self, *processing_elements):
        """
        Initializes the pipeline.

        :*processing_elements: A list of processing elements, i.e. elements
            that comply with the ProcessingElement interface.
        """
        self._elements = processing_elements

    def get_output(self, input_):
        """
        Computes the output of the pipeline, given the input from the environment.

        :input_: The input coming from the environment
        :returns: The output of the last processing element
        """
        x = input_

        for el in self._elements:
            x = el.get_output(x)

        return x

    def set_reward(self, reward):
        """
        Sets the reward for all the processing elements of the pipeline

        :reward: A scalar value for the reward
        """
        for e in self._elements:
            e.set_reward(reward)

    def new_episode(self):
        """
        Initializes a new episode for the whole pipeline
        """
        for e in self._elements:
            e.new_episode()

    def get_elements(self):
        return self._elements

    def set_elements(self, value):
        self._elements = value

    def get_element(self, index):
        """
        Retrieves the i-th element from the pipeline

        :index: The index of the element
        :returns: The index-th processing element
        """
        return self._elements[index]

    def set_elements(self, index, element):
        """
        Replaces the index-th element of the pipeline with a new one

        :index: The index of the element
        :element: The new element
        """
        self._elements[index] = element

    def __str__(self):
        return repr(self)

    def __repr__(self):
        string_repr = ""

        for e in self._elements:
            string_repr += e.__class__.__name__
            string_repr += "\n"
            string_repr += repr(e)
            string_repr += "\n"
        return string_repr


class PipelineFactory:
    """
    Interface for factories of pipelines.
    """
    @abstractmethod
    def ask_pop(self):
        """
        Produces and returns a population of pipelines.
        :returns: A population of pipelines.
        """
        pass

    @abstractmethod
    def tell_pop(self, pipelines, fitnesses):
        """
        Sets the fitnesses for the population of pipelines.

        :pipelines: A list of pipelines
        :fitnesses: A list of fitnesses (list of floats)
        """
        pass


class CartesianProductPipelineFactory(PipelineFactory, metaclass=PipelineFactoryMetaClass):

    """
    This class implements a factory that produces a number of
    individuals equal to the number of different combinations that
    we can perform with all the populations of each optimizer.
    """

    def __init__(self, element_factories, config):
        """
        Initializes the factory.

        :element_factories: A list of factories, one for each
            processing element that will compose the pipeline
        :config: A dictionary containing at least the following parameters:
            - aggregation_fcn: (list of) string that represent the aggregation function
                that must be used when assigning the fitnesses to the individuals of
                the single populations.
                Must be either "min", "mean", "max", "median".
                If this parameter is a single string, then that measure will be used for
                all the factories.
                Otherwise, if a list of strings is passed, each factory will use its own
                aggregation function.
        """
        PipelineFactory.__init__(self)

        self._element_factories = element_factories
        self._aggregation_fcn = config["aggregation_fcn"]
        if isinstance(self._aggregation_fcn, str):
            self._aggregation_fcn = [self._aggregation_fcn] * len(self._element_factories)
        assert len(self._aggregation_fcn) == len(self._element_factories), \
                "The length of the aggregation functions must be equal to the" + \
                " len of the element factories"
        self._last_indices = None

    def _mix_populations(self, populations):
        output = []
        indices = []

        if len(populations) > 1:
            subpop, mixed_indices = self._mix_populations(populations[1:])
        else:
            subpop = [[]]
            mixed_indices = [[]]

        for ind_idx, ind in enumerate(populations[0]):
            for pop, partial_idx in zip(subpop, mixed_indices):
                output.append([ind, *pop])
                indices.append([ind_idx, *partial_idx])

        indices = np.array(indices)

        return output, indices

    def ask_pop(self):
        """
        Produces and returns a population of pipelines.

        :returns: A population of pipelines.
        """
        populations = []

        for factory in self._element_factories:
            populations.append(factory.ask_pop())

        mixed_population, self._last_indices = self._mix_populations(populations)

        mixed_population = [Pipeline(*x) for x in mixed_population]
        return mixed_population

    def _hook(self, pipelines, fitnesses, pop_idx, factory):
        pass

    def tell_pop(self, pipelines, fitnesses):
        """
        Sets the fitnesses for the population of pipelines.

        :pipelines: A list of pipelines
        :fitnesses: A list of fitnesses (list of floats)
        """
        assert self._last_indices is not None, \
                "tell_pop cannot be called before ask_pop"

        for pop_idx, factory in enumerate(self._element_factories):
            cur_fitnesses = []
            pop_size = max(self._last_indices[:, pop_idx]) + 1
            for ind_idx in range(pop_size):
                aggr_fcn = getattr(np, self._aggregation_fcn[pop_idx])
                selected_indices = np.where(self._last_indices[:, pop_idx] == ind_idx)[0]
                aggr_fit = aggr_fcn(fitnesses[selected_indices])
                cur_fitnesses.append(aggr_fit)
            cur_fitnesses = np.array(cur_fitnesses)

            self._hook(pipelines, fitnesses, pop_idx, factory)
            factory.tell_pop(cur_fitnesses)
