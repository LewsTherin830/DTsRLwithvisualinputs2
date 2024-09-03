#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    src.trees
    ~~~~~~~~~

    This module implements the conditions to be used in the decision trees

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import abc
from .nodes import Node


class Condition(Node):
    """
    This is the base class for the conditions.
    """
    BRANCH_LEFT = True
    BRANCH_RIGHT = False

    def __init__(self, left, right):
        """
        Initializes an internal node (that checks a condition)

        :left: The node in the left branch (i.e. the one taken when the
               condition evaluates to True)
        :right: The node in the right branch (i.e. the one taken when
               the condition evaluates to False)

        """
        Node.__init__(self)

        self._left = left
        self._right = right

    def get_left(self):
        return self._left

    def get_right(self):
        return self._right

    def set_left(self, value):
        self._left = value

    def set_right(self, value):
        self._right = value

    @abc.abstractstaticmethod
    def get_trainable_parameters(self):
        """
        Returns a list of parameters with their type
        (input_index, int or float) as a string.
        """
        pass

    @abc.abstractmethod
    def set_params_from_list(self, params):
        """
        Sets its parameters according to the parameters specified by
        the input list.

        :params: A list of params (int or float)
        """
        pass

    def get_output(self, input_):
        """
        Computes the output associated to its inputs (i.e. computes
        the path of the input vector (or vectors) in the tree and returns
        the decision associated to it).

        :input_: A 1D numpy array
        :returns: A 1D numpy array
        """
        assert len(input_.shape) == 1, "Only 1D arrays are currently supported"
        if self.get_branch(input_) == Condition.BRANCH_LEFT:
            return self._left.get_output(input_)
        else:
            return self._right.get_output(input_)

    @abc.abstractmethod
    def get_branch(self, inputs):
        """
        Computes the branch taken given the inputs

        :inputs: 1D numpy array (1 sample) or 2D numpy array (N samples)
        :returns: A numpy array where each element is either:
            - True: Left branch has been taken
            - False: Right branch has been taken
        """
        pass

    def empty_buffers(self):
        self._left.empty_buffers()
        self._right.empty_buffers()

    def copy(self):
        """
        Returns a copy of itself
        """
        new = Condition(self.get_left().copy(), self.get_right().copy())
        return new


class OrthogonalCondition(Condition):
    """
    This class implements orthogonal conditions for the decision tree.
    Orthogonal conditions are the ones that generate hyperplanes that are
    orthogonal to the chosen axis (i.e. they test only one variable).
    """

    def __init__(self, feature_idx, split_value, left=None, right=None):
        """
        Initializes the condition.

        :feature_idx: the index of the variable (that has to be tested) in
                 the input vector
        :split_value: The value used for the test
        :left: The left node. Default: None.
        :right: The right node. Default: None.
        """
        Condition.__init__(self, left, right)

        self._feature_idx = feature_idx
        self._split_value = split_value

    def get_branch(self, inputs):
        """
        Computes the branch taken given the inputs

        :inputs: 1D numpy array (1 sample) or 2D numpy array (N samples)
        :returns: A numpy array where each element is either:
            - True: Left branch has been taken
            - False: Right branch has been taken
        """
        if len(inputs.shape) == 1:
            return inputs[self._feature_idx] < self._split_value
        return inputs[:, self._feature_idx] < self._split_value

    @staticmethod
    def get_trainable_parameters():
        """
        Returns a list of parameters with their type
        (input_index, int or float) as a string.
        """
        return ["input_index", "float"]

    @staticmethod
    def check_params(params):
        """
        Checks whether the parameters are good for this type of node.
        If not, it raises an AssertionError

        :params: A list of params (int or float)
        """
        assert len(params) >= 2, \
            "This type of condition requires 2 parameters."

    @staticmethod
    def create_from_params(params):
        """
        Creates a condition from its parameters

        :params: A list of params (int or float)
        """
        OrthogonalCondition.check_params(params)
        return OrthogonalCondition(int(params[0]), float(params[1]))

    def set_params_from_list(self, params):
        """
        Sets its parameters according to the parameters specified by
        the input list.

        :params: A list of params (int or float)
        """
        OrthogonalCondition.check_params(params)
        self._feature_idx = int(params[0])
        self._split_value = float(params[1])

    def get_feature_idx(self):
        return self._feature_idx

    def get_split_value(self):
        return self._split_value

    def set_feature_idx(self, value):
        self._feature_idx = value

    def set_split_value(self, value):
        self._split_value = value

    def __str__(self):
        return f"x_{self._feature_idx} < {self._split_value}"

    def copy(self):
        """
        Returns a copy of itself
        """
        new = OrthogonalCondition(
            self.get_feature_idx(),
            self.get_split_value(),
            self.get_left().copy(),
            self.get_right().copy()
        )
        return new


class ConditionFactory:
    """
    A factory for conditions
    """
    ORTHOGONAL = "orthogonal"

    CONDITIONS = {
        ORTHOGONAL: OrthogonalCondition,
    }

    def __init__(self, condition_type="orthogonal"):
        """
        Initializes the factory of conditions

        :condition_type: strings supported:
            - orthogonal
        """
        self._condition_type = condition_type

    def create(self, params):
        """
        Creates a condition
        :returns: A Condition
        """
        return self.CONDITIONS[self._condition_type].create_from_params(params)

    def get_trainable_parameters(self):
        """
        Returns a list of parameters with their type (int or float).
        """
        return self.CONDITIONS[self._condition_type].get_trainable_parameters()


