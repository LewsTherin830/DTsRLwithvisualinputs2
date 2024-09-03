#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    cv_modules.convolution
    ~~~~~~~~~~~~~~~~~~~~~~

    This module implement processing elements for CV tasks

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import torch
import numpy as np
from cv_modules import CVMetaClass, CVProcessingElement


class ConvolutionModule(CVProcessingElement, metaclass=CVMetaClass):
    """
    This class implement a convolution module composed of
    several convolutionary filters.
    This module, when called, returns a list of (x, y) coordinates
    with length equal to the number of filters, where each coordinate
    is the point of the image that returned the maximum convolution score.
    """

    def __init__(self, filters, device):
        """
        Initializes a new ConvolutionModule.

        :filters: A list of filters (i.e., list (or ndarray) of 3D ndarrays)
        :device: The device where to perform the operations
        """
        CVProcessingElement.__init__(self)
        self._device = device
        self._filters = torch.Tensor(np.array(filters))
        self._filters = self._filters.to(self._device)

    def get_output(self, input_):
        """
        This method returns the output of the agent given the input

        :input_: The agent's input (i.e., a 3d ndarray)
        :returns: The agent's output, which may be either a scalar, an ndarray
            or a torch Tensor
        """
        output = []

        X = torch.Tensor(input_)
        X = X.to(self._device)
        if len(X.shape) == 3:
            X = X.view(1, *X.shape)
        conv_output = torch.nn.functional.conv2d(
            X, self._filters).detach().cpu().numpy()[0]
        im_w = X.shape[-2]

        for out in conv_output:
            argmax = np.argmax(out.flatten())
            x = argmax % im_w
            y = argmax // im_w

            output.extend([x, y])

            # Leave a hook for further customization
            self._hook(input_, x, y)

        output = np.array(output)
        return output

    def _hook(self, input_, x, y):
        """
        This method is a hook method for allowing further
        customization

        :input_: The input
        :x: The x of the most important patch
        :y: The y of the most important patch
        :returns: TODO

        """
        pass

    @staticmethod
    def _get_filter_sizes(filter_sizes, n_filters):
        if isinstance(filter_sizes, int):
            filter_sizes = [[filter_sizes] * 2] * n_filters
        return filter_sizes

    @staticmethod
    def get_n_params(config):
        """
        Computes the number of parameters that are needed
        to produce the processing element

        :config: A dictionary containing at least the following parameters:
            - n_filters: The number of filters
            - filter_sizes: A list of (width, height) for each filter
        :returns: An int
        """
        filter_sizes = ConvolutionModule._get_filter_sizes(
            config["filter_sizes"]
        )

        acc = 0
        for w, h in filter_sizes:
            acc += w * h * 3
        return acc

    @staticmethod
    def from_params(params, config):
        """
        Creates a new ConvolutionModule from a list of parameters and
        a dictionary

        :params: An ndarray of numerical values
        :config: A dictionary containing the information to build the module
        :returns: An instance of CVProcessingElement
        """
        filter_sizes = ConvolutionModule._get_filter_sizes(
            config["filter_sizes"], config["n_filters"]
        )
        device = config.get("device", "cpu")

        filters = []

        cur_i = 0
        for w, h in filter_sizes:
            new_filter = params[cur_i:cur_i + w * h * 3]
            new_filter = new_filter.reshape(3, h, w)
            filters.append(new_filter)

        return ConvolutionModule(filters, device)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self._filters)
