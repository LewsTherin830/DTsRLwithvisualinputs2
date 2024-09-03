#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    src.common
    ~~~~~~~~~~

    This module contains common utilities for optimizers

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""


class OptMetaClass(type):
    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        OptMetaClass._registry[cls.__name__] = cls
        return cls

    @staticmethod
    def get(class_name):
        """
        Retrieves the class associated to the string

        :class_name: The name of the class
        :returns: A class
        """
        return OptMetaClass._registry[class_name]


class ContinuousOptimizationMetaClass(type):
    _registry = {}

    def __init__(cls, clsname, bases, methods):
        super().__init__(clsname, bases, methods)
        ContinuousOptimizationMetaClass._registry[cls.__name__] = cls

    @staticmethod
    def get(class_name):
        """
        Retrieves the class associated to the string

        :class_name: The name of the class
        :returns: A class
        """
        return ContinuousOptimizationMetaClass._registry[class_name]

