#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    tasks.task
    ~~~~~~~~~~

    This module implements an interface for the tasks

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
from abc import abstractmethod


class Task:
    """
    This is the base class for tasks
    """

    @abstractmethod
    def seed(self, seed):
        """
        Set the seed for the random number generator in the task

        :seed: The seed
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the current environment
        :returns: A new observation
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Sets the action/prediction for the current sample and
        advances the simulation to the next step

        :action: The action/prediction from the RL model
        :returns: A tuple (next observation, reward,
            final step for the simulation)
        """
        pass

    @abstractmethod
    def close(self):
        """
        Deallocates the environment
        """
        pass

    @abstractmethod
    def sample_action(self):
        """
        Samples an action from the action space
        """
        pass
