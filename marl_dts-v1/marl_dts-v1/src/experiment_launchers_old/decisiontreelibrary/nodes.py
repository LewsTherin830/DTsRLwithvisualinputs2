#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    src.nodes
    ~~~~~~~~~

    This module contains the implementations of the node base class

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import abc


class Node:
    """
    A basic node for a decision tree
    """

    @abc.abstractmethod
    def get_output(self, input_features):
        """
        Computes the output of the current node,
        possibly calling the child leaves

        :input_features: An array containing all the input features
        :returns: A numpy array whose last dimension has size
                  equal to the size of the output space

        """
        pass

    @abc.abstractmethod
    def empty_buffers(self):
        """
        Clears the buffers of the node and, recursively,
        the buffers of all its subtree
        """
        pass

    @abc.abstractmethod
    def copy(self):
        """
        Returns a copy of itself
        """
        pass

    @staticmethod
    def get_trainable_parameters():
        """
        Returns a list of parameters with their type.
        The type must not necessarily be a python primitive,
        but should indicate the type of the trainable parameters.
        """
        return []
