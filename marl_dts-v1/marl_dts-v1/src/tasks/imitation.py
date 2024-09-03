#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    tasks.imitation
    ~~~~~~~~~~

    This module implements an imitation task

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import numpy as np
from typing import List
from .common import TaskMetaClass


class ImitationTask(metaclass=TaskMetaClass):
    """
    This class implements an imitation task
    """
    def __init__(self, inputs: List, outputs: List, sampling_prob=1):
        """
        Initializes the task

        :inputs: A list of paths to npy files containing the inputs
        :outputs: A list of paths to npy files with the reference's outputs
        :sampling_prob: The proportion of samples to sample from the reference
        """
        self._inputs = []
        self._outputs = []

        for i, o in zip(inputs, outputs):
            # Use outputs instead of inputs, they are smaller in size
            cur_outputs = np.load(o)
            n = round(len(cur_outputs) * sampling_prob)

            sampling_indices = np.random.choice(np.arange(len(cur_outputs)), n)
            if len(sampling_indices) > 0:
                cur_inputs = np.load(i)[sampling_indices]
                cur_outputs = cur_outputs[sampling_indices]

                self._inputs.extend(cur_inputs)
                self._outputs.extend(cur_outputs)

        self._inputs = np.array(self._inputs)
        self._outputs = np.array(self._outputs)
        self.reset()

    def seed(self, seed):
        """
        Set the seed for the random number generator in the task

        :seed: The seed
        """
        pass

    def reset(self):
        """
        Resets the current environment
        :returns: A new observation
        """
        self._ctr = 0
        return self._inputs[self._ctr]

    def step(self, action):
        """
        Sets the action/prediction for the current sample and
        advances the simulation to the next step

        :action: The action/prediction from the RL model
        :returns: A tuple (next observation, reward,
            final step for the simulation)
        """
        if hasattr(action, "__len__"):
            # Is an iterable
            action = np.argmax(action)
        if action == self._outputs[self._ctr]:
            reward = 1/len(self._outputs)
        else:
            reward = 0
        self._ctr += 1
        if self._ctr < len(self._outputs):
            new_sample = self._inputs[self._ctr]
            done = False
        else:
            done = True
            new_sample = None
        return new_sample, reward, done

    def close(self):
        """
        Deallocates the environment
        """
        del self._inputs
        del self._outputs
        del self._ctr


class VectorizedImitationTask(ImitationTask, metaclass=TaskMetaClass):
    """
    This class implements a vectorized imitation task
    """
    def __init__(self, inputs: List, outputs: List, sampling_prob=1):
        """
        Initializes the task

        :inputs: A list of paths to npy files containing the inputs
        :outputs: A list of paths to npy files with the reference's outputs
        :sampling_prob: The proportion of samples to sample from the reference
        """
        super(VectorizedImitationTask, self).__init__(
            inputs, outputs, sampling_prob
        )
        self._ctr = 0

    def reset(self):
        """
        Resets the current environment
        :returns: A new observation
        """
        return self._inputs

    def step(self, action):
        """
        Sets the action/prediction for the current sample and
        advances the simulation to the next step

        :action: The action/prediction from the RL model
        :returns: A tuple (next observation, reward,
            final step for the simulation)
        """
        if hasattr(action, "__len__"):
            action = np.argmax(action, axis=-1)
        reward = np.mean(action == self._outputs)
        done = True
        new_sample = None
        return new_sample, reward, done
