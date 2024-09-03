#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    src.test_conditions
    ~~~~~~~~~~~~~~~~~~~

    Tests for the condition module

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import pytest
import numpy as np
from numpy.linalg import norm
from ..leaves import QLearningLeaf
from ..conditions import Condition, OrthogonalCondition, ConditionFactory


def test_c():
    l1 = QLearningLeaf(10)
    l2 = QLearningLeaf(10)
    c = Condition(l1, l2)
    assert l1 is not l2
    assert c.get_left() is not c.get_right()
    assert c.get_left() is l1
    assert c.get_right() is l2
    c.set_left(l2)
    assert c.get_left() is l2
    assert c.get_left() is c.get_right()
    c.set_right(l1)
    assert c.get_right() is l1
    assert c.get_left() is not c.get_right()
    return c


inputs = np.random.uniform(-100, 100, (10, 10, 3))
lots_of_inputs = np.random.uniform(-100, 100, (10, 100, 3))


@pytest.mark.parametrize("inputs", inputs)
def test_eb(inputs):
    l1 = QLearningLeaf(10)
    l2 = QLearningLeaf(10)
    c = Condition(l1, l2)
    for i in inputs:
        l1.get_output(i)
        l2.get_output(i)
    s1, a1, r1 = c.get_left().get_buffers()
    s2, a2, r2 = c.get_right().get_buffers()
    assert len(s1) > 0
    assert len(a1) > 0
    assert len(s2) > 0
    assert len(a2) > 0
    c.empty_buffers()
    s1, a1, r1 = c.get_left().get_buffers()
    s2, a2, r2 = c.get_right().get_buffers()
    assert len(s1) == 0
    assert len(a1) == 0
    assert len(s2) == 0
    assert len(a2) == 0


def test_cc():
    c = test_c()
    new = c.copy()

    assert c is not new
    assert c.get_left() is not new.get_left()
    assert c.get_right() is not new.get_right()


#########################
#  OrthogonalCondition  #
#########################


@pytest.mark.parametrize("inputs", inputs)
def test_ocgo1(inputs):
    c = OrthogonalCondition(0, 0, QLearningLeaf(2), QLearningLeaf(3))

    for i in inputs:
        c.get_output(i)
        if i[0] < 0:
            assert norm(c.get_left().get_buffers()[0][-1] - i) == 0
        else:
            assert norm(c.get_right().get_buffers()[0][-1] - i) == 0


@pytest.mark.parametrize("inputs", lots_of_inputs)
def test_ocgo2(inputs):
    l1 = QLearningLeaf(10)
    l2 = QLearningLeaf(10)
    l3 = QLearningLeaf(10)
    l4 = QLearningLeaf(10)

    left = OrthogonalCondition(1, 0, l1, l2)
    right = OrthogonalCondition(2, 0, l3, l4)

    root = OrthogonalCondition(0, 0, left, right)

    for i in inputs:
        root.get_output(i)
        if i[0] < 0:
            if i[1] < 0:
                assert norm(root.get_left().get_left().get_buffers()[0][-1] - i) == 0
            else:
                assert norm(root.get_left().get_right().get_buffers()[0][-1] - i) == 0
        else:
            if i[2] < 0:
                assert norm(root.get_right().get_left().get_buffers()[0][-1] - i) == 0
            else:
                assert norm(root.get_right().get_right().get_buffers()[0][-1] - i) == 0


def test_ocgtp():
    c = OrthogonalCondition(0, 1, None, None)
    assert c.get_trainable_parameters()[0] == "input_index"
    assert c.get_trainable_parameters()[1] == "float"


def test_occ():
    c = OrthogonalCondition(0, 0, QLearningLeaf(10), QLearningLeaf(10))
    new = c.copy()

    assert c is not new
    assert new.get_feature_idx() == c.get_feature_idx()
    assert new.get_split_value() == c.get_split_value()
    assert c.get_left() is not new.get_left()
    assert c.get_right() is not new.get_right()

    new.set_params_from_list([1, 1])
    assert new.get_feature_idx() != c.get_feature_idx()
    assert new.get_split_value() != c.get_split_value()
    assert c.get_left() is not new.get_left()
    assert c.get_right() is not new.get_right()


######################
#  ConditionFactory  #
######################

def test_cf():
    cf = ConditionFactory("orthogonal")
    c = cf.create([0, 1])
    assert isinstance(c, OrthogonalCondition)
    assert c.get_feature_idx() == 0
    assert c.get_split_value() == 1.0
