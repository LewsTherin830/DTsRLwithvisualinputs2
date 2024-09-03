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
# import torch
import numpy as np
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
        # assert len(input_.shape) == 1, "Only 1D arrays are currently supported"
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

    def get_code(self):
        return f"if input_[{self._feature_idx}] < {self._split_value}:"

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

    @classmethod
    def create_from_params(cls, params):
        """
        Creates a condition from its parameters

        :params: A list of params (int or float)
        """
        cls.check_params(params)
        return cls(int(params[0]), float(params[1]))

    def set_params_from_list(self, params):
        """
        Sets its parameters according to the parameters specified by
        the input list.

        :params: A list of params (int or float)
        """
        self.check_params(params)
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
        new = self.__class__(
            self.get_feature_idx(),
            self.get_split_value(),
            self.get_left().copy(),
            self.get_right().copy()
        )
        return new


#######################################################################
#                      Differentiable conditions                      #
#######################################################################


class DifferentiableOrthogonalCondition(OrthogonalCondition):
    """
    A differentiable condition that can be trained by backprop
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
        OrthogonalCondition.__init__(self, feature_idx, split_value, left, right)

        self._feature_idx = feature_idx
        self._split_value = torch.tensor([split_value], requires_grad=True)

    def get_output(self, input_):
        """
        Computes the output associated to its inputs (i.e. computes
        the path of the input vector (or vectors) in the tree and returns
        the decision associated to it).

        :input_: A 1D numpy array
        :returns: A 1D numpy array
        """
        s = self.get_coefficient(input_)
        return s * self._left.get_output(input_) + (1 - s) * self._right.get_output(input_)

    def get_coefficient(self, input_):
        return 1 - torch.sigmoid(input_[self._feature_idx] - self._split_value)

    def get_params(self):
        return self._split_value

    def discretize(self):
        return OrthogonalCondition(self._feature_idx,
                                   self._split_value,
                                   self._left.discretize(),
                                   self._right.discretize())


#######################################################################
#                         Oblique conditions                          #
#######################################################################

class Oblique2Condition(Condition):
    """
    This class implements orthogonal conditions for the decision tree.
    Orthogonal conditions are the ones that generate hyperplanes that are
    orthogonal to the chosen axis (i.e. they test only one variable).
    """

    def __init__(self, feature_1, feature_2, left=None, right=None):
        """
        Initializes the condition.

        :feature_1: the index of the variable (that has to be tested) in
                 the input vector
        :feature_2: the index of the variable (that has to be tested) in
                 the input vector
        :left: The left node. Default: None.
        :right: The right node. Default: None.
        """
        Condition.__init__(self, left, right)

        self._feature_1 = feature_1
        self._feature_2 = feature_2

    def get_branch(self, inputs):
        """
        Computes the branch taken given the inputs

        :inputs: 1D numpy array (1 sample) or 2D numpy array (N samples)
        :returns: A numpy array where each element is either:
            - True: Left branch has been taken
            - False: Right branch has been taken
        """
        if len(inputs.shape) == 1:
            return inputs[self._feature_1] < inputs[self._feature_2]
        return inputs[:, self._feature_1] < inputs[:, self._feature_2]

    def get_code(self):
        return f"if input_[{self._feature_1}] < input_[{self._feature_2}]:"

    @staticmethod
    def get_trainable_parameters():
        """
        Returns a list of parameters with their type
        (input_index, int or float) as a string.
        """
        return ["input_index", "input_index"]

    @staticmethod
    def check_params(params):
        """
        Checks whether the parameters are good for this type of node.
        If not, it raises an AssertionError

        :params: A list of params (int or float)
        """
        assert len(params) >= 2, \
            "This type of condition requires 2 parameters."

    @classmethod
    def create_from_params(cls, params):
        """
        Creates a condition from its parameters

        :params: A list of params (int or float)
        """
        cls.check_params(params)
        return cls(int(params[0]), int(params[1]))

    def set_params_from_list(self, params):
        """
        Sets its parameters according to the parameters specified by
        the input list.

        :params: A list of params (int or float)
        """
        self.check_params(params)
        self._feature_1 = int(params[0])
        self._feature_2 = int(params[1])

    def get_feature_1(self):
        return self._feature_1

    def get_feature_2(self):
        return self._feature_2

    def set_feature_1(self, value):
        self._feature_1 = value

    def set_feature_2(self, value):
        self._feature_2 = value

    def __str__(self):
        return f"x_{self._feature_1} < x_{self._feature_2}"

    def copy(self):
        """
        Returns a copy of itself
        """
        new = self.__class__(
            self.get_feature_1(),
            self.get_feature_2(),
            self.get_left().copy(),
            self.get_right().copy()
        )
        return new


class Oblique2ConditionWithOffset(Condition):
    """
    This class implements orthogonal conditions for the decision tree.
    Orthogonal conditions are the ones that generate hyperplanes that are
    orthogonal to the chosen axis (i.e. they test only one variable).
    """

    def __init__(self, feature_1, feature_2, offset, sign1, sign2, left=None, right=None):
        """
        Initializes the condition.

        :feature_1: the index of the variable (that has to be tested) in
                 the input vector
        :feature_2: the index of the variable (that has to be tested) in
                 the input vector
        :left: The left node. Default: None.
        :right: The right node. Default: None.
        """
        Condition.__init__(self, left, right)

        self._feature_1 = feature_1
        self._feature_2 = feature_2
        self._offset = offset
        self._sign1 = sign1
        self._sign2 = sign2

    def get_branch(self, inputs):
        """
        Computes the branch taken given the inputs

        :inputs: 1D numpy array (1 sample) or 2D numpy array (N samples)
        :returns: A numpy array where each element is either:
            - True: Left branch has been taken
            - False: Right branch has been taken
        """
        if len(inputs.shape) == 1:
            return self._sign1 * inputs[self._feature_1] < self._sign2 * inputs[self._feature_2] + self._offset
        raise NotImplemented

    def get_code(self):
        return f"if input_[{self._feature_1}] < input_[{self._feature_2}]:"

    @staticmethod
    def get_trainable_parameters():
        """
        Returns a list of parameters with their type
        (input_index, int or float) as a string.
        """
        return ["input_index", "input_index", "float", "sign", "sign"]

    @staticmethod
    def check_params(params):
        """
        Checks whether the parameters are good for this type of node.
        If not, it raises an AssertionError

        :params: A list of params (int or float)
        """
        assert len(params) >= 5, \
            "This type of condition requires 5 parameters."

    @staticmethod
    def sign(x):
        return -1 if x < 0 else 1

    @classmethod
    def create_from_params(cls, params):
        """
        Creates a condition from its parameters

        :params: A list of params (int or float)
        """
        cls.check_params(params)
        return cls(int(params[0]), int(params[1]), float(params[2]), cls.sign(params[3]), cls.sign(params[4]))

    def set_params_from_list(self, params):
        """
        Sets its parameters according to the parameters specified by
        the input list.

        :params: A list of params (int or float)
        """
        self.check_params(params)
        self._feature_1 = int(params[0])
        self._feature_2 = int(params[1])
        self._offset = float(params[2])
        self._sign1 = self.sign(params[3])
        self._sign2 = self.sign(params[4])

    def get_feature_1(self):
        return self._feature_1

    def get_feature_2(self):
        return self._feature_2

    def set_feature_1(self, value):
        self._feature_1 = value

    def set_feature_2(self, value):
        self._feature_2 = value

    def get_offset(self):
        return self._offset

    def set_offset(self, value):
        self._offset = value

    def get_sign1(self):
        return self._sign1

    def set_sign1(self, value):
        self._sign1 = self.sign(value)

    def get_sign2(self):
        return self._sign2

    def set_sign2(self, value):
        self._sign2 = self.sign(value)

    def __str__(self):
        return f"{self._sign1} * x_{self._feature_1} < {self._sign2} * x_{self._feature_2} + {self._offset}"

    def copy(self):
        """
        Returns a copy of itself
        """
        new = self.__class__(
            self.get_feature_1(),
            self.get_feature_2(),
            self.get_offset(),
            self.get_sign1(),
            self.get_sign2(),
            self.get_left().copy(),
            self.get_right().copy()
        )
        return new


class ObliqueCondition(Condition):
    """
    This class implements orthogonal conditions for the decision tree.
    Orthogonal conditions are the ones that generate hyperplanes that are
    orthogonal to the chosen axis (i.e. they test only one variable).
    """

    def __init__(self, params, left=None, right=None):
        """
        Initializes the condition.

        :feature_1: the index of the variable (that has to be tested) in
                 the input vector
        :feature_2: the index of the variable (that has to be tested) in
                 the input vector
        :left: The left node. Default: None.
        :right: The right node. Default: None.
        """
        Condition.__init__(self, left, right)

        self._params = params

    def get_branch(self, inputs):
        """
        Computes the branch taken given the inputs

        :inputs: 1D numpy array (1 sample) or 2D numpy array (N samples)
        :returns: A numpy array where each element is either:
            - True: Left branch has been taken
            - False: Right branch has been taken
        """
        if len(inputs.shape) == 1:
            inp = np.array([*inputs, 1])
            output = np.dot(inp, self._params)
            return output < 0
        raise NotImplemented

    def get_code(self):
        return f"if <x, {self._params}>:"

    @staticmethod
    def get_trainable_parameters():
        """
        Returns a list of parameters with their type
        (input_index, int or float) as a string.
        """
        return ["float"]

    @staticmethod
    def check_params(params):
        """
        Checks whether the parameters are good for this type of node.
        If not, it raises an AssertionError

        :params: A list of params (int or float)
        """

    @classmethod
    def create_from_params(cls, params):
        """
        Creates a condition from its parameters

        :params: A list of params (int or float)
        """
        return cls(params)

    def set_params_from_list(self, params):
        """
        Sets its parameters according to the parameters specified by
        the input list.

        :params: A list of params (int or float)
        """
        self._params = params

    def get_params(self):
        return self._params

    def set_params(self, value):
        self._params = value

    def __str__(self):
        return f"<x, {self._params}>"

    def copy(self):
        """
        Returns a copy of itself
        """
        new = self.__class__(
            self.get_params().copy(),
            self.get_left().copy(),
            self.get_right().copy()
        )
        return new


#######################################################################
#                               Factory                               #
#######################################################################


class ConditionFactory:
    """
    A factory for conditions
    """
    ORTHOGONAL = "orthogonal"
    DIFFERENTIABLE = "differentiable"
    OBLIQUE2 = "2vars"
    OBLIQUE2OFFSET = "2varswoffset"
    OBLIQUE = "oblique"

    CONDITIONS = {
        ORTHOGONAL: OrthogonalCondition,
        DIFFERENTIABLE: DifferentiableOrthogonalCondition,
        OBLIQUE2: Oblique2Condition,
        OBLIQUE2OFFSET: Oblique2ConditionWithOffset,
        OBLIQUE: ObliqueCondition
    }

    def __init__(self, condition_type="orthogonal", n_inputs=None):
        """
        Initializes the factory of conditions

        :condition_type: strings supported:
            - orthogonal
            - differentiable
            - 2vars
            - 2varswoffset
            - oblique
        :n_inputs: only needed for the condition_type "oblique".
        """
        self._condition_type = condition_type
        self._n_inputs = n_inputs

    def create(self, params):
        """
        Creates a condition
        :returns: A Condition
        """
        cond = self.CONDITIONS[self._condition_type].create_from_params(params)
        return cond

    def get_trainable_parameters(self):
        """
        Returns a list of parameters with their type (int or float).
        """
        param_types = self.CONDITIONS[self._condition_type].get_trainable_parameters()
        if self._condition_type == self.OBLIQUE:
            return param_types * (self._n_inputs + 1)
        return param_types


