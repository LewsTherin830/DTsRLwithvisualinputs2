# -*- coding: utf-8 -*-
"""
    src.__init__
    ~~~~~~~~~~~~

    This module contains processing elements that do not
    need training.

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import os
import cv2
import pickle
import numpy as np
from processing_element import ProcessingElement, ProcessingElementFactory, PEFMetaClass


class UtilMetaClass(type):
    _registry = {}

    def __init__(cls, clsname, bases, methods):
        super().__init__(clsname, bases, methods)
        UtilMetaClass._registry[cls.__name__] = cls

    @staticmethod
    def get(class_name):
        """
        Retrieves the class associated to the string

        :class_name: The name of the class
        :returns: A class
        """
        return UtilMetaClass._registry[class_name]


class Resizer(ProcessingElement, metaclass=UtilMetaClass):
    """
    This processing element handles the resizing of images.
    """

    def __init__(self, width, height):
        """
        Initializes the processing element

        :width: The final width
        :height: The final height

        """
        ProcessingElement.__init__(self)

        self._width = width
        self._height = height

    def get_output(self, input_):
        """
        Resizes the image

        :input_: The image to resize
        :returns: The resized image
        """
        if len(input_.shape) > 3:
            # Multiple samples
            return np.array(
                [cv2.resize(x, (self._width, self._height)) for x in input_]
            )
        return cv2.resize(input_, (self._width, self._height))

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"Resizer ({self._width}, {self._height})"


class MoveDim(ProcessingElement, metaclass=UtilMetaClass):

    """
    This processing element moves the dimensions in an tensor
    """

    def __init__(self, orig_dim, dest_dim):
        """
        Initializes the processing element

        :orig_dim: The origin of the dimension to move
        :dest_dim: The destination of the dimension
        """
        ProcessingElement.__init__(self)

        self._orig_dim = orig_dim
        self._dest_dim = dest_dim

    def get_output(self, input_):
        """
        Moves the exis

        :input_: The tensor
        :returns: The tensor with moved dimensions
        """
        return np.moveaxis(input_, self._orig_dim, self._dest_dim)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"MoveDim ({self._orig_dim}, {self._dest_dim})"


class Cropper(ProcessingElement, metaclass=UtilMetaClass):
    """
    This processing element handles the cropping of images
    """

    def __init__(self, left=None, top=None, right=None, bottom=None):
        """
        Initializes the processing element

        :left: The left margin. Default: None.
        :top: The top margin. Default: None.
        :right: The right margin. Default: None.
        :bottom: The bottom margin. Default: None.
        """
        ProcessingElement.__init__(self)

        self._left = left
        self._top = top
        self._right = right
        self._bottom = bottom

    def _set_if_not_none(self, var, default):
        if var is not None:
            if var <= 0:
                return default - var
            return var
        return default

    def get_output(self, input_):
        """
        Crops the image.

        :input_: The image to crop
        :returns: The cropped image
        """
        l = self._set_if_not_none(self._left, 0)
        t = self._set_if_not_none(self._top, 0)
        r = self._set_if_not_none(self._right, input_.shape[1])
        b = self._set_if_not_none(self._bottom, input_.shape[0])

        if len(input_.shape) > 3:
            # Multiple samples
            return np.array([x[t:b, l:r] for x in input_])
        return input_[t:b, l:r]

    def __repr__(self):
        return f"Cropper {self._left, self._top, self._right, self._bottom}"

    def __str__(self):
        return repr(self)

    def __unicode__(self):
        return repr(self)


class MinMaxNormalizer(ProcessingElement, metaclass=UtilMetaClass):

    """
    This processing element normalizes all the elements by using
    min-max normalization.
    """

    def __init__(self, min_, max_):
        """
        Initializes the processing element

        :min_: The minimum used to normalize
        :max_: The maximum used to normalize

        """
        ProcessingElement.__init__(self)

        self._min = min_
        self._max = max_

    def get_output(self, input_):
        """
        Normalizes the input

        :input_: The input
        :returns: The normalized input
        """
        return (np.array(input_) - self._min) / (self._max - self._min)

    def __repr__(self):
        return f"Normalizer ({self._min}, {self._max})"

    def __str__(self):
        repr(self)

    def __unicode__(self):
        repr(self)


class OneHotEncoder(ProcessingElement, metaclass=UtilMetaClass):
    """
    This processing element encodes the input with one-hot encoding
    """

    def __init__(self, n_actions):
        """
        Initializes the processing element

        :n_actions: The number of actions
        """
        ProcessingElement.__init__(self)

        self._n_actions = n_actions

    def get_output(self, input_):
        """
        Encodes the input

        :input_: The input (a scalar)
        :returns: One-hot representation of the input
        """
        if not isinstance(input_, int):
            # Several actions
            output = np.zeros(len(input_), self._n_actions)
            for i, j in enumerate(input_):
                output[i, j] = 1
        else:
            output = np.zeros(self._n_actions, dtype=int)
            output[input_] = 1
        return output


class LinearModel(ProcessingElement, metaclass=UtilMetaClass):
    """
    This processing element Predicts the future for each feature,
    memorizing the past inputs
    """

    def __init__(self, predictors_path, window_size):
        """
        Initializes the processing element

        :predictors_path: The path to the dir containing the models
        :window_size: The size of the window containing the past values used to
        predict the future. "1" means that only the current value is used.
        """
        ProcessingElement.__init__(self)

        self._predictors = []
        files = list(os.listdir(predictors_path))
        files = sorted(files)
        files = list(map(lambda f: os.path.join(predictors_path, f), files))

        for p in files:
            self._predictors.append(
                pickle.load(open(p, "rb"))
            )
        self._window_size = window_size
        self._memory = [[] for _ in range(len(predictors_path))]

    def get_output(self, input_):
        """
        Encodes the input

        :input_: The input
        :returns: The prediction for each feature in the next step
        """
        output = []
        for i, (p, m) in enumerate(zip(self._predictors, self._memory)):
            self._memory[i].append(input_[i])
            m = self._memory[i]
            if len(m) < self._window_size:
                self._memory[i] = [0] * (self._window_size - len(m)) + m
            output.append(p.predict([self._memory[i]]))
        return output


class UtilFactory(ProcessingElementFactory, metaclass=PEFMetaClass):
    """
    This factory simply returns utility processing elements
    """

    def __init__(self, module, **module_args):
        """
        Initializes the factory

        :module: The name of the class to produce
        :kwargs: all the parameters needed by the processing element
        """
        ProcessingElementFactory.__init__(self)
        self._module = module
        self._module_args = module_args

    def ask_pop(self):
        """
        Returns the utility processing element
        """
        return [UtilMetaClass.get(self._module)(**self._module_args)]

    def tell_pop(self, fitnesses):
        """
        This function does nothing.
        We keep it for compatibility with the interface.
        """
        # We do not need to tell the fitnesses
        pass
