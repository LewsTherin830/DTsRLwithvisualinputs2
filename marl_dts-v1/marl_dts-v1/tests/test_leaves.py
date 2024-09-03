#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    src.test_leaves
    ~~~~~~~~~

    Tests for the leaves module

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import pytest
import numpy as np
from functools import partial
from ..leaves import Leaf, QLearningLeaf, RandomInitQLearningLeafDecorator, \
        EpsilonGreedyQLearningLeafDecorator, NoBuffersDecorator, \
        QLearningLeafFactory, QLambdaDecorator


#####################
#  Leaf Base class  #
#####################

n_tests = 10

inputs = [
    np.random.uniform(
        -100, 100, size=np.random.randint(1, 100)
    ) for _ in range(n_tests)
]

leaves = [
    Leaf,
    partial(QLearningLeaf, n_actions=10, learning_rate=0.1),
]

actions = np.random.randint(0, 10, n_tests)

rewards = np.random.uniform(-10, 10, n_tests)

lgo_inputs_to_test = []
lra_inputs_to_test = []
lsr_inputs_to_test = []

for inp in inputs:
    for leaf in leaves:
        lgo_inputs_to_test.append((inp, leaf))
        for a in actions:
            lra_inputs_to_test.append((inp, a, leaf))
            for r in rewards:
                lsr_inputs_to_test.append((inp, a, r, leaf))


@pytest.mark.parametrize("input_, leaf_type", lgo_inputs_to_test)
def test_lgo(input_, leaf_type):
    leaf = leaf_type()
    leaf.get_output(input_)
    assert len(leaf.get_inputs()) == 1
    assert np.linalg.norm(leaf.get_inputs()[0] - input_) == 0
    states, actions, rewards = leaf.get_buffers()
    assert len(states) == 1
    assert np.linalg.norm(states[0] - input_) == 0
    if not isinstance(leaf, Leaf):
        assert len(actions) == 1
    assert len(rewards) == 0
    return leaf


@pytest.mark.parametrize(
    "input_, action, reward, leaf_type",
    lsr_inputs_to_test
)
def leaf_lsrgo(input_, action, reward, leaf_type):
    leaf = test_lgo(input_, leaf_type)
    leaf.set_reward(reward)
    assert len(leaf._rewards) == 1
    assert leaf._rewards[0] == reward
    states, actions, rewards = leaf.get_buffers()
    assert len(states) == 1
    if not isinstance(leaf, Leaf):
        assert len(actions) == 1
    assert len(rewards) == 1
    assert rewards[0] == reward
    return leaf


@pytest.mark.parametrize(
    "input_, action, reward, leaf_type",
    lsr_inputs_to_test
)
def test_lebgo(input_, action, reward, leaf_type):
    leaf = leaf_lsrgo(input_, action, reward, leaf_type)
    leaf.empty_buffers()
    assert len(leaf.get_inputs()) == 0
    assert len(leaf._action_history) == 0
    assert len(leaf._rewards) == 0
    states, actions, rewards = leaf.get_buffers()
    assert len(states) == 0
    assert len(actions) == 0
    assert len(rewards) == 0


@pytest.mark.parametrize("leaf_type", leaves)
def test_lc(leaf_type):
    leaf = leaf_type()
    new_leaf = leaf.copy()

    assert id(leaf) != id(new_leaf)
    for attribute in dir(leaf):
        if attribute[:2] != "__":
            if not isinstance(attribute, str) and \
               not isinstance(attribute, int):
                assert id(getattr(leaf, attribute)) != \
                        id(getattr(new_leaf, attribute))

###################
#  QLearningLeaf  #
###################


n_actions = 10
bounds = [-10, 10]
eps = 0.05

decorators = [
    None,
    partial(RandomInitQLearningLeafDecorator, low=bounds[0], high=bounds[1]),
    partial(EpsilonGreedyQLearningLeafDecorator, epsilon=eps, decay=1, min_epsilon=eps),
    partial(NoBuffersDecorator),
]

actions = np.random.randint(0, n_actions, 20)

qllgo_inputs = []
qllfa_inputs = []

for d in decorators:
    for i in inputs:
        qllgo_inputs.append((i, d))
        for a in actions:
            qllfa_inputs.append((i, a, d))


def make_qlearningleaf(decorator):
    leaf = QLearningLeaf(n_actions, None)
    if decorator is not None:
        leaf = decorator(leaf)
    return leaf


@pytest.mark.parametrize("decorator", decorators)
def test_qlliq(decorator):
    leaf = make_qlearningleaf(decorator)
    leaf._init_q(n_actions)
    assert len(leaf.get_q()) == n_actions
    if isinstance(leaf, RandomInitQLearningLeafDecorator):
        assert max(leaf.get_q()) <= bounds[1]
        assert min(leaf.get_q()) >= bounds[0]
    else:
        assert max(leaf.get_q()) == 0
        assert min(leaf.get_q()) == 0
    return leaf


@pytest.mark.parametrize("input_, decorator", qllgo_inputs)
def test_qllgo(input_, decorator):
    leaf = make_qlearningleaf(decorator)
    outputs = np.array([leaf.get_output(input_) for i in range(101)])

    if isinstance(leaf, RandomInitQLearningLeafDecorator):
        # The action should always be the same
        # For the moment let's ignore the case in which two elements
        #   have the same Q values
        assert max(outputs) == min(outputs)
    elif isinstance(leaf, EpsilonGreedyQLearningLeafDecorator):
        # Since epsilon is low, this should be the most frequent choice
        median = np.median(outputs)
        # Give this some margin
        assert np.sum(outputs == median) > 1 - 2 * eps
    # Default Leaf Behavior
    assert max(outputs) < n_actions
    assert min(outputs) >= 0


@pytest.mark.parametrize("input_, decorator", qllgo_inputs)
def test_qllsr(input_, decorator):
    leaf = make_qlearningleaf(decorator)
    for i in np.arange(0, 100):
        action = leaf.get_output(input_)
        rew = np.random.uniform(-1, 1)
        cur_q = leaf.get_q()[action]
        leaf.set_reward(rew)
        if rew < cur_q:
            assert leaf.get_q()[action] <= cur_q
        else:
            assert leaf.get_q()[action] >= cur_q


@pytest.mark.parametrize("input_, action, leaf_type", qllfa_inputs)
def test_qllfa(input_, action, leaf_type):
    leaf = make_qlearningleaf(leaf_type)
    leaf.force_action(input_, action)
    assert len(leaf.get_inputs()) == 1
    assert np.linalg.norm(leaf.get_inputs()[0] - input_) == 0
    states, actions, rewards = leaf.get_buffers()
    assert len(states) == 1
    assert np.linalg.norm(states[0] - input_) == 0
    assert len(actions) == 1
    assert actions[0] == action
    assert len(rewards) == 0
    return leaf


@pytest.mark.parametrize("decorator", decorators)
def test_qllc(decorator):
    leaf = make_qlearningleaf(decorator)
    new_leaf = leaf.copy()

    leaf._q[0] = 1
    assert np.linalg.norm(leaf._q - new_leaf._q) != 0


decorator_list = [
    [],
    [partial(RandomInitQLearningLeafDecorator, low=bounds[0], high=bounds[1])],
    [partial(EpsilonGreedyQLearningLeafDecorator, epsilon=eps, decay=1, min_epsilon=eps)],
    [partial(NoBuffersDecorator)],
    [partial(QLambdaDecorator, decay=0)],
    [
        partial(RandomInitQLearningLeafDecorator, low=bounds[0], high=bounds[1]),
        partial(EpsilonGreedyQLearningLeafDecorator, epsilon=eps, decay=1, min_epsilon=eps)
    ],
    [
        partial(RandomInitQLearningLeafDecorator, low=bounds[0], high=bounds[1]),
        partial(NoBuffersDecorator)
    ],
    [
        partial(RandomInitQLearningLeafDecorator, low=bounds[0], high=bounds[1]),
        partial(EpsilonGreedyQLearningLeafDecorator, epsilon=eps, decay=1, min_epsilon=eps),
        partial(NoBuffersDecorator)
    ],
    [
        partial(RandomInitQLearningLeafDecorator, low=bounds[0], high=bounds[1]),
        partial(EpsilonGreedyQLearningLeafDecorator, epsilon=eps, decay=1, min_epsilon=eps),
        partial(QLambdaDecorator, decay=0)
    ],
    [
        partial(RandomInitQLearningLeafDecorator, low=bounds[0], high=bounds[1]),
        partial(NoBuffersDecorator),
        partial(QLambdaDecorator, decay=0)
    ],
    [
        partial(RandomInitQLearningLeafDecorator, low=bounds[0], high=bounds[1]),
        partial(EpsilonGreedyQLearningLeafDecorator, epsilon=eps, decay=1, min_epsilon=eps),
        partial(NoBuffersDecorator),
        partial(QLambdaDecorator, decay=0)
    ],
]


combinations_ds = []

for d in decorator_list:
    for i in range(10):
        combinations_ds.append((d, i))


@pytest.mark.parametrize("decorators, seed", combinations_ds)
def test_qll0(decorators, seed):
    np.random.seed(seed)

    leaf = QLearningLeaf(1, None)
    for decorator in decorators:
        leaf = decorator(leaf)

    rewards = []

    n_steps = 1000
    for i in range(n_steps):
        out = leaf.get_output(None)
        assert out == 0
        rew = np.random.uniform()
        leaf.set_reward(rew)
        rewards.append(rew)
        """
        q = leaf.get_q()
        if not np.isnan(q[0]):
            print(q)
        """

    q = leaf.get_q()
    mean = 0
    expected = 0
    qlambda = 0
    for i in range(n_steps):
        lr = 1 / (i + 1)
        mean = mean * (i / (i + 1)) + 1/(i + 1) * rewards[i]
        expected = expected * (1 - lr) + lr * rewards[i]
        delta = rewards[i] - qlambda
        qlambda = qlambda + lr * delta
    print(mean, np.mean(rewards), expected, qlambda)
    assert min(mean/q[0], q[0]/mean) > 0.99


@pytest.mark.parametrize("decorators, seed", combinations_ds)
def test_qll1(decorators, seed):
    np.random.seed(seed)
    a = 0.1

    leaf = QLearningLeaf(1, a)
    for decorator in decorators:
        leaf = decorator(leaf)

    rewards = []
    n_steps = 1000

    q0 = leaf.get_q()[0]
    for i in range(n_steps):
        out = leaf.get_output(None)
        assert out == 0
        rew = np.random.uniform()
        leaf.set_reward(rew)
        rewards.append(rew)
        """
        q = leaf.get_q()
        if not np.isnan(q[0]):
            print(q)
        """

    q = leaf.get_q()
    movingavg = 0
    movingavg += ((1-a) ** (n_steps)) * q0
    # print(movingavg, q0, ((1-a) ** n_steps))
    for i in range(1, n_steps):
        movingavg += a * ((1 - a) ** (n_steps - 1 - i)) * rewards[i]
        # print(movingavg, rewards[i], a * ((1 - a) ** (n_steps - i - 1)))
    print(len(rewards), rewards[-20:])
    assert min(movingavg/q[0], q[0]/movingavg) > 0.999

#################
#  LeafFactory  #
#################


decorators_list = [
    None,
    [["RandomInit", {"low": -100, "high": 100}]],
    [["EpsilonGreedy", {"epsilon": 0.05, "decay": 1}]],
    [["NoBuffers", {}]],
    [
        ["RandomInit", {"low": -100, "high": 100}],
        ["EpsilonGreedy", {"epsilon": 0.05, "decay": 1}]
    ],
    [
        ["RandomInit", {"low": -100, "high": 100}],
        ["EpsilonGreedy", {"epsilon": 0.05, "decay": 1}],
        ["NoBuffers", {}]
    ],
]


@pytest.mark.parametrize("decorator", decorators_list)
def test_qllf(decorator):
    d = [] if decorator is None else decorator
    lf = QLearningLeafFactory({"n_actions": 10}, d).create()
    assert isinstance(lf, Leaf)
    assert isinstance(lf, QLearningLeaf)
    if decorator is not None:
        assert isinstance(lf._leaf, QLearningLeaf)
        assert lf.get_q() is lf._leaf.get_q()
    return lf


@pytest.mark.parametrize("decorator", decorators_list)
def test_qllfc(decorator):
    leaf = test_qllf(decorator)
    new_leaf = leaf.copy()

    assert id(leaf) != id(new_leaf)
    for attribute in dir(leaf):
        if attribute[:2] != "__":
            if not isinstance(attribute, str) and \
               not isinstance(attribute, int):
                assert id(getattr(leaf, attribute)) != \
                        id(getattr(new_leaf, attribute))


