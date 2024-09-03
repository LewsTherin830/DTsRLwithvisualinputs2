#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    src.test_trees
    ~~~~~~~~~~~~~~

    Tests for the trees module

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import pytest
import numpy as np
from numpy.linalg import norm
from ..trees import DecisionTree, RLDecisionTree
from ..leaves import QLearningLeafFactory
from ..conditions import ConditionFactory

##################
#  DecisionTree  #
##################

inputs = np.random.uniform(-100, 100, (10, 100, 5))
types = [
    DecisionTree,
    RLDecisionTree
]

full_inputs = []


for t in types:
    for i in inputs:
        full_inputs.append((t, i))


def test_dt():
    c = ConditionFactory().create([1, 1])
    d = ConditionFactory().create([0, 0])
    dt = DecisionTree(c)

    assert dt.get_root() is c
    assert dt.get_root().get_feature_idx() == c.get_feature_idx()
    assert dt.get_root().get_split_value() == c.get_split_value()

    dt.set_root(d)
    assert dt.get_root() is d
    assert dt.get_root().get_feature_idx() == d.get_feature_idx()
    assert dt.get_root().get_split_value() == d.get_split_value()


@pytest.mark.parametrize("tp, inputs", full_inputs)
def test_dtfull(tp, inputs):
    root = ConditionFactory().create([0, 0])
    left = ConditionFactory().create([1, 1])
    right = ConditionFactory().create([2, 1])
    qllf = QLearningLeafFactory({"n_actions": 10}, [])
    l1 = qllf.create()
    l2 = qllf.create()
    l3 = qllf.create()
    l4 = qllf.create()

    root.set_left(left)
    root.set_right(right)
    left.set_left(l1)
    left.set_right(l2)
    right.set_left(l3)
    right.set_right(l4)

    if tp == RLDecisionTree:
        dt = RLDecisionTree(root, 0)
    else:
        dt = DecisionTree(root)

    leaves = dt.get_leaves()
    assert len(leaves) == 4
    assert l1 in leaves
    assert l2 in leaves
    assert l3 in leaves
    assert l4 in leaves

    for i in inputs:
        dt.get_output(i)
        if i[0] < 0:
            # Left
            if i[1] < 1:
                assert norm(l1.get_buffers()[0][-1] - i) == 0
            else:
                assert norm(l2.get_buffers()[0][-1] - i) == 0
        else:
            if i[2] < 1:
                assert norm(l3.get_buffers()[0][-1] - i) == 0
            else:
                assert norm(l4.get_buffers()[0][-1] - i) == 0

    s1, a1, _ = l1.get_buffers()
    s2, a2, _ = l2.get_buffers()
    s3, a3, _ = l3.get_buffers()
    s4, a4, _ = l4.get_buffers()

    assert len(s1) > 0
    assert len(s2) > 0
    assert len(s3) > 0
    assert len(s4) > 0
    assert len(a1) > 0
    assert len(a2) > 0
    assert len(a3) > 0
    assert len(a4) > 0

    new_dt = dt.copy()

    new_leaves = new_dt.get_leaves()

    assert l1 not in new_leaves
    assert l2 not in new_leaves
    assert l3 not in new_leaves
    assert l4 not in new_leaves

    l1, l2, l3, l4 = new_leaves

    assert len(s1) > 0
    assert len(s2) > 0
    assert len(s3) > 0
    assert len(s4) > 0
    assert len(a1) > 0
    assert len(a2) > 0
    assert len(a3) > 0
    assert len(a4) > 0

    dt.empty_buffers()

    s1, a1, _ = l1.get_buffers()
    s2, a2, _ = l2.get_buffers()
    s3, a3, _ = l3.get_buffers()
    s4, a4, _ = l4.get_buffers()

    assert len(s1) == 0
    assert len(s2) == 0
    assert len(s3) == 0
    assert len(s4) == 0
    assert len(a1) == 0
    assert len(a2) == 0
    assert len(a3) == 0
    assert len(a4) == 0

    l1, l2, l3, l4 = new_leaves

    assert len(s1) == 0
    assert len(s2) == 0
    assert len(s3) == 0
    assert len(s4) == 0
    assert len(a1) == 0
    assert len(a2) == 0
    assert len(a3) == 0
    assert len(a4) == 0


####################
#  RLDecisionTree  #
####################


@pytest.mark.parametrize("inputs", inputs)
def test_rldtsr(inputs):
    root = ConditionFactory().create([0, 0])
    left = ConditionFactory().create([1, 1])
    right = ConditionFactory().create([2, 1])
    qllf = QLearningLeafFactory({"n_actions": 10, "learning_rate": 0.1}, [])
    l1 = qllf.create()
    l2 = qllf.create()
    l3 = qllf.create()
    l4 = qllf.create()

    root.set_left(left)
    root.set_right(right)
    left.set_left(l1)
    left.set_right(l2)
    right.set_left(l3)
    right.set_right(l4)

    dt = RLDecisionTree(root, 0.9)

    for i in inputs:
        if i[0] < 0:
            if i[1] < 1:
                tgt_leaf = l1
            else:
                tgt_leaf = l2
        else:
            if i[2] < 1:
                tgt_leaf = l3
            else:
                tgt_leaf = l4

        dt.empty_buffers()
        output = dt.get_output(i)
        previous_q = tgt_leaf.get_q().copy()
        rew = np.random.uniform(-1, 1)
        dt.set_reward(rew)
        dt.get_output(i)
        dt.set_reward(rew)
        cur_q = tgt_leaf.get_q()

        assert cur_q[output] == 0.9 * previous_q[output] + 0.1 * (rew + 0.9 * max(previous_q))
        for j in range(10):
            if j == output:
                continue
            assert cur_q[j] == previous_q[j]


@pytest.mark.parametrize("inputs", inputs)
def test_rldtsrfa(inputs):
    root = ConditionFactory().create([0, 0])
    left = ConditionFactory().create([1, 1])
    right = ConditionFactory().create([2, 1])
    qllf = QLearningLeafFactory({"n_actions": 10, "learning_rate": 0.1}, [])
    l1 = qllf.create()
    l2 = qllf.create()
    l3 = qllf.create()
    l4 = qllf.create()

    root.set_left(left)
    root.set_right(right)
    left.set_left(l1)
    left.set_right(l2)
    right.set_left(l3)
    right.set_right(l4)

    dt = RLDecisionTree(root, 0.9)

    for i in inputs:
        if i[0] < 0:
            if i[1] < 1:
                tgt_leaf = l1
            else:
                tgt_leaf = l2
        else:
            if i[2] < 1:
                tgt_leaf = l3
            else:
                tgt_leaf = l4

        dt.empty_buffers()
        output = dt.force_action(i, 0)
        assert output == 0
        previous_q = tgt_leaf.get_q().copy()
        rew = np.random.uniform(-1, 1)
        dt.set_reward(rew)
        dt.get_output(i)
        dt.set_reward(rew)
        cur_q = tgt_leaf.get_q()

        assert cur_q[output] == 0.9 * previous_q[output] + 0.1 * (rew + 0.9 * max(previous_q))
        for j in range(10):
            if j == output:
                continue
            assert cur_q[j] == previous_q[j]


@pytest.mark.parametrize("inputs", inputs)
def test_rldtsree(inputs):
    root = ConditionFactory().create([0, 0])
    left = ConditionFactory().create([1, 1])
    right = ConditionFactory().create([2, 1])
    qllf = QLearningLeafFactory({"n_actions": 10, "learning_rate": 0.1}, [])
    l1 = qllf.create()
    l2 = qllf.create()
    l3 = qllf.create()
    l4 = qllf.create()

    root.set_left(left)
    root.set_right(right)
    left.set_left(l1)
    left.set_right(l2)
    right.set_left(l3)
    right.set_right(l4)

    dt = RLDecisionTree(root, 0.9)

    for i in inputs:
        if i[0] < 0:
            if i[1] < 1:
                tgt_leaf = l1
            else:
                tgt_leaf = l2
        else:
            if i[2] < 1:
                tgt_leaf = l3
            else:
                tgt_leaf = l4

        dt.empty_buffers()
        output = dt.get_output(i)
        print("Output:", output)

        s, a, r = tgt_leaf.get_buffers()
        assert len(s) == 1
        assert len(a) == 1
        assert len(r) == 0

        previous_q = tgt_leaf.get_q().copy()
        rew = np.random.uniform(-1, 1)
        dt.set_reward(rew)
        dt.set_reward_end_of_episode()
        cur_q = tgt_leaf.get_q()

        assert cur_q[output] == 0.9 * previous_q[output] + 0.1 * (rew)
        for j in range(10):
            if j == output:
                continue
            assert cur_q[j] == previous_q[j]
