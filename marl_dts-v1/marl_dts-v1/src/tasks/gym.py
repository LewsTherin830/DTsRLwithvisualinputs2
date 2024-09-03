#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    tasks.gym
    ~~~~~~~~~~

    This module implements a wrapper for OpenAI Gym

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import gym
import numpy as np
from .common import TaskMetaClass


class GymTask(metaclass=TaskMetaClass):
    """
    This class represents a wrapper for OpenAI Gym environments
    """
    def __init__(self, env_name, **env_kwargs):
        self._env = gym.make(env_name, **env_kwargs)

    def seed(self, seed):
        """
        Set the seed for the random number generator in the task

        :seed: The seed
        """
        self._env.seed(seed)

    def reset(self):
        """
        Resets the current environment
        :returns: A new observation
        """
        return self._env.reset()

    def step(self, action):
        """
        Sets the action/prediction for the current sample and
        advances the simulation to the next step

        :action: The action/prediction from the RL model
        :returns: A tuple (next observation, reward,
            final step for the simulation)
        """
        new_action = action
        if hasattr(self._env.action_space, "n"):
            # Discrete-action environment
            if hasattr(action, "__len__"):
                # The action is a list
                new_action = np.argmax(action)
        return self._env.step(new_action)[:3]

    def close(self):
        """
        Deallocates the environment
        """
        self._env.close()

    def sample_action(self):
        """
        Samples an action from the action space
        """
        return self._env.action_space.sample()
