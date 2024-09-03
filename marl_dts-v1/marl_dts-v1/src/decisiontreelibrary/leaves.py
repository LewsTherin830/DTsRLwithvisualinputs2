#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    src.leaves
    ~~~~~~~~~~

    This module implements the leaves that can be used in the trees
    and the factories to build them.

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import abc
import numpy as np
from .nodes import Node
from copy import deepcopy


class Leaf(Node):
    """
    This is the base class for the leaves, defines their interface.
    """

    def __init__(self):
        """
        Initializes a leaf.
        """
        # Call to the super
        Node.__init__(self)
        # Initialize the buffers
        self.empty_buffers()

    def get_output(self, input_):
        """
        Computes the output for the given input

        :input_: An array of input features
        :returns: A numpy array whose last dimension has the size equal to
        the dimensionality of the actions
        """
        self._inputs.append(input_)  # Record the input
        # Return an invalid action. The base class cannot compute decisions
        return np.zeros(1)

    def record_action(self, action):
        """
        Records an action into the history

        :action: The action taken by the leaf
        """
        self._action_history.append(action)

    def set_reward(self, reward):
        """
        Gives the reward to the leaf.

        :reward: The total reward given to the leaf (e.g.
                 for Q-learning it should be reward = r + gamma · Qmax(s'))
        """
        n_in = len(self._inputs)
        """
        assert len(self._rewards) in [n_in - 2, n_in - 1], \
            f"The number of calls to get_output are different \
from the number of calls to set_reward ({len(self._rewards)}, \
{len(self._inputs)})."
        """
        self._rewards.append(reward)

    def empty_buffers(self):
        """
        Deletes the buffers associated to the leaf.
        """
        self._inputs = []
        self._action_history = []
        self._rewards = []

    @abc.abstractmethod
    def get_value(self):
        """
        Returns a generic value associated to the leaf to compute the reward
        that should be given to the other leaves
        (e.g. for Q-learning it should be max(Q))
        """
        pass

    def get_buffers(self):
        """
        Returns 3 lists:
            - History of inputs
            - History of actions
            - History of rewards
        """
        return self._inputs, self._action_history, self._rewards

    def set_buffers(self, inputs, actions, rewards):
        """
        Sets the three buffers as the leaves' buffers
        """
        self._inputs = inputs
        self._action_history = actions
        self._rewards = rewards

    def copy(self):
        """
        Returns a copy of itself
        """
        return Leaf()

    def get_inputs(self):
        return self._inputs


class QLearningLeaf(Leaf):
    """
    This class implements a leaf that learns the state-action mapping
    by means of Q-learning.
    """

    def __init__(self, n_actions, learning_rate=None):
        """
        Initializes the leaf

        :n_actions: An integer that specifies the number of available actions
        :learning_rate: A float (or None) that specifies the learning rate.
                        If it is None, a schedule of learning rate of 1/x,
                        where x is the number of visits to that action,
                        will be used

        """
        Leaf.__init__(self)

        self._learning_rate = learning_rate
        self._q = self._init_q(n_actions)
        self._visits = np.zeros(n_actions)

    def __str__(self):
        return f"{np.argmax(self._q)} ({np.sum(self._visits)} visits)"

    def _init_q(self, n_actions):
        """
        Initializes the Q function (to zero)

        :n_actions: The number of available actions
        :returns: A list representing the multi-armed bandit version of the
        Q-function
        """
        return np.zeros(n_actions)

    def get_output(self, input_):
        """
        Computes the output of the leaf

        :input_: An array of input features
        :returns: A numpy array whose last dimension has the size equal to
        the dimensionality of the actions
        """
        super().get_output(input_)
        self._last_action = np.argmax(self._q)
        self._visits[self._last_action] += 1
        super().record_action(self._last_action)
        return self._last_action

    def set_reward(self, reward):
        """
        Gives the reward to the leaf.

        :reward: The total reward given to the leaf (e.g.
                 for Q-learning it should be reward = r + gamma · Qmax(s'))
        """
        super().set_reward(reward)
        # Take the last action without reward
        last_action = self._last_action
        if self._learning_rate is None:
            lr = 1 / self._visits[last_action]
        else:
            lr = self._learning_rate

        old_q = self._q[last_action]
        self._q[last_action] = (1 - lr) * old_q + lr * reward

    def get_value(self):
        """
        Returns a generic value associated to the leaf to compute the reward
        that should be given to the other leaves
        (e.g. for Q-learning it should be max(Q))
        """
        return np.max(self._q)

    def get_q(self):
        """
        Returns the current Q function
        """
        return self._q

    def force_action(self, input_, action):
        """
        This method makes the leaf "return" a given action, i.e. it allows the
        leaf to take a decision taken by someone else (e.g. useful for
        exploration strategies)

        :input_: An array of input features
        :action: The action "forced" by someone else
        """
        super().get_output(input_)
        self._last_action = action
        self._visits[self._last_action] += 1
        super().record_action(self._last_action)
        return self._last_action

    def get_n_actions(self):
        """
        Returns the number of actions available to the leaf.
        """
        return len(self._q)

    def set_q(self, q):
        """
        Sets the Q function

        :q: A list of Q-values
        """
        assert len(q) == len(self._q), \
            "The new Q has to be of the same size as the old Q"
        self._q = q

    def get_lr(self):
        return self._learning_rate

    def get_visits(self):
        return self._visits

    def set_visits(self, value):
        self._visits = value

    def copy(self):
        """
        Returns a copy of itself
        """
        if self._learning_rate is not None:
            lr = type(self._learning_rate)(self._learning_rate)
        else:
            lr = None
        leaf = QLearningLeaf(self.get_n_actions(), lr)
        leaf.set_q(self.get_q().copy())
        leaf.set_visits(deepcopy(self.get_visits()))
        return leaf


class QLearningLeafDecorator(QLearningLeaf):
    """
    A base class for leaf decorators
    """

    def __init__(self, leaf):
        """
        Initializes the base decorator

        :leaf: An instance of Leaf
        """
        self._leaf = leaf
        QLearningLeaf.__init__(
            self,
            leaf.get_n_actions(),
            leaf.get_lr()
        )

    def __str__(self):
        return str(self._leaf)

    def record_action(self, action):
        """
        Records an action into the history

        :action: The action taken by the leaf
        """
        self._leaf.record_action(action)

    def empty_buffers(self):
        """
        Deletes the buffers associated to the leaf.
        """
        self._leaf.empty_buffers()

    def get_buffers(self):
        """
        Returns 3 lists:
            - History of inputs
            - History of actions
            - History of rewards
        """
        return self._leaf.get_buffers()

    def set_buffers(self, inputs, actions, rewards):
        """
        Sets the three buffers as the leaves' buffers
        """
        return self._leaf.set_buffers(inputs, actions, rewards)

    def get_output(self, input_):
        """
        Computes the output of the leaf

        :input_: An array of input features
        :returns: A numpy array whose last dimension has the size equal to
        the dimensionality of the actions
        """
        return self._leaf.get_output(input_)

    def set_reward(self, reward):
        """
        Gives the reward to the leaf.

        :reward: The total reward given to the leaf (e.g.
                 for Q-learning it should be reward = r + gamma · Qmax(s'))
        """
        self._leaf.set_reward(reward)

    def get_value(self):
        """
        Returns a generic value associated to the leaf to compute the reward
        that should be given to the other leaves
        (e.g. for Q-learning it should be max(Q))
        """
        return self._leaf.get_value()

    def get_q(self):
        """
        Returns the current Q function
        """
        return self._leaf.get_q()

    def force_action(self, input_, action):
        """
        This method makes the leaf "return" a given action, i.e. it allows the
        leaf to take a decision taken by someone else (e.g. useful for
        exploration strategies)

        :input_: An array of input features
        :action: The action "forced" by someone else
        """
        return self._leaf.force_action(input_, action)

    def get_n_actions(self):
        """
        Returns the number of actions available to the leaf.
        """
        return self._leaf.get_n_actions()

    def set_q(self, q):
        """
        Sets the Q function

        :q: A list of Q-values
        """
        self._leaf.set_q(q)

    def get_lr(self):
        return self._leaf.get_lr()

    def get_visits(self):
        return self._leaf.get_visits()

    def copy(self):
        """
        Returns a copy of itself
        """
        return QLearningLeafDecorator(self._leaf.copy())

    def get_inputs(self):
        return self._leaf.get_inputs()


class EpsilonGreedyQLearningLeafDecorator(QLearningLeafDecorator):
    """
    QLearningLeafDecorator that allows a QLearningLeaf (or an extending class)
    to have an epsilon-greedy exploration strategy
    """

    def __init__(self, leaf, epsilon=None, decay=1, min_epsilon=0):
        """
        Initializes the decorator

        :leaf: An instance of QLearningLeaf
        :epsilon: A float indicating the (initial) probability of exploration. Default: 1/k
        :decay: Optional. A float indicating the decay factor for epsilon.
                Default: 1 (No decay)
        :min_epsilon: Optional. The minimum value of epsilon.
                Default: 0 (No min value)
        """
        QLearningLeafDecorator.__init__(self, leaf)
        self._epsilon = epsilon
        self._decay = decay
        self._min_epsilon = min_epsilon
        self._visits = 0

    def get_output(self, input_):
        """
        Computes the output of the leaf

        :input_: An array of input features
        :returns: A numpy array whose last dimension has the size equal to
        the dimensionality of the actions
        """
        self._visits += 1
        if self._epsilon is None:
            eps = 1/self._visits
        else:
            self._epsilon *= self._decay
            eps = max(self._epsilon, self._min_epsilon)

        if np.random.uniform() < eps:
            return self._leaf.force_action(
                input_,
                np.random.randint(0, self._leaf.get_n_actions())
            )
        else:
            return self._leaf.get_output(input_)

    def copy(self):
        """
        Returns a copy of itself
        """
        new = EpsilonGreedyQLearningLeafDecorator(
            self._leaf.copy(),
            self._epsilon,
            self._decay,
            self._min_epsilon
        )
        return new


class RandomInitQLearningLeafDecorator(QLearningLeafDecorator):
    """
    A decorator that allows to randomly intialize the Q function
    """

    def __init__(self, leaf, low, high, distribution="uniform"):
        """
        Initializes the decorator

        :leaf: An instance of QLearningLeaf
        :low: The low bound for the initial Q function.
        :high: The upper bound for the initial Q function
        :distribution: The name of the distribution.
                  Can be either "normal" or "uniform".
                  In case the distribution is "normal", low and high will
                  be used to compute mean and std deviation of the normal.

        """
        QLearningLeafDecorator.__init__(self, leaf)

        n = self._leaf.get_n_actions()
        self._distribution = distribution
        self._low = low
        self._high = high
        if distribution == "normal":
            mean = np.mean([low, high])
            std = np.std([low, high])
            self._leaf.set_q(np.random.normal(mean, std, n))
        else:
            self._leaf.set_q(np.random.uniform(low, high, n))

    def copy(self):
        """
        Returns a copy of itself
        """
        new = RandomInitQLearningLeafDecorator(
            self._leaf.copy(),
            self._low,
            self._high,
            self._distribution
        )
        new.set_q(self._leaf.get_q())
        return new


class NoBuffersDecorator(QLearningLeafDecorator):
    """
    A decorator that allows to avoid memory leaks due to the big number
    of transitions recorded. Useful when a lot of trees are used with
    algorithms that do not need their history (e.g. evolutionary algorithms)
    """

    def __init__(self, leaf):
        """
        Initializes the decorator

        :leaf: An instance of QLearningLeaf
        """
        QLearningLeafDecorator.__init__(self, leaf)

    def get_output(self, input_):
        """
        Returns the output associated with the input

        :input_: The input vector (1D)
        :returns: A numpy array whose last dimension has the size equal to
        the dimensionality of the actions
        """
        # Retrieve the output from the leaf
        out = self._leaf.get_output(input_)
        # Delete the unnecessary elements in the buffers
        # I.e. the ones whose reward has already been set
        inputs, actions, rewards = self._leaf.get_buffers()
        unnecessary = len(rewards) - 1
        if unnecessary > 0:
            del inputs[:unnecessary]
            del rewards[:unnecessary]
            del actions[:unnecessary]
            self._leaf.set_buffers(inputs, actions, rewards)
        # return the output of the leaf
        return out

    def copy(self):
        """
        Returns a copy of itself
        """
        new = NoBuffersDecorator(
            self._leaf.copy()
        )
        return new


class QLambdaDecorator(QLearningLeafDecorator):
    """
    A decorator that implements Naive Q(lambda).
    """

    def __init__(self, leaf, decay):
        """
        Initializes the decorator

        :leaf: The leaf
        :decay: The decay factor: lambda * gamma
        """
        QLearningLeafDecorator.__init__(self, leaf)
        self._decay = decay

        self._eligibility_traces = np.zeros(len(self._leaf.get_q()))
        self._visits = np.zeros(len(self._eligibility_traces))
        self._last_action = None

    def copy(self):
        """
        Returns a copy of itself
        """
        return QLambdaDecorator(self._leaf.copy(), self._decay)

    def get_output(self, input_):
        """
        Computes the output of the leaf

        :input_: An array of input features
        :returns: A numpy array whose last dimension has the size equal to
        the dimensionality of the actions
        """
        output = self._leaf.get_output(input_)
        self._last_action = output
        self._eligibility_traces[self._last_action] += 1
        return output

    def set_reward(self, reward):
        """
        Gives the reward to the leaf.

        :reward: The total reward given to the leaf (e.g.
                 for Q-learning it should be reward = r + gamma · Qmax(s'))
        """
        q = self._leaf.get_q()

        for i in range(len(self._eligibility_traces)):
            _learning_rate = self._leaf.get_lr()
            self._visits[i] += 1
            if _learning_rate is None:
                lr = 1 / self._visits[i]
            else:
                lr = _learning_rate

            delta = reward - q[i]
            q[i] += lr * self._eligibility_traces[i] * delta

            self._eligibility_traces[i] *= self._decay

        self._leaf.set_q(q)

    def empty_buffers(self):
        self._leaf.empty_buffers()
        self._eligibility_traces = np.zeros(len(self._leaf.get_q()))


class QLearningLeafFactory:
    """
    A base class for the factories of leaves.
    """

    DECORATOR_DICT = {
        "EpsilonGreedy": EpsilonGreedyQLearningLeafDecorator,
        "RandomInit": RandomInitQLearningLeafDecorator,
        "NoBuffers": NoBuffersDecorator,
        "QLambda": QLambdaDecorator,
    }

    def __init__(self, leaf_params, decorators):
        """
        Initializes the factory

        :leaf_params: A dictionary containing all the parameters of the leaf
        :decorators: A list of (decorator_name, **params)
        """
        self._leaf_params = leaf_params
        self._decorators = decorators
        for name, _ in self._decorators:
            assert name in self.DECORATOR_DICT, \
                    f"Unable to find the decorator {name}\n\
                    Available decorators: {self.DECORATOR_DICT.keys()}"

    def create(self):
        """ Creates a leaf and returns it """
        leaf = QLearningLeaf(**self._leaf_params)
        for name, params in self._decorators:
            leaf = self.DECORATOR_DICT[name](leaf, **params)
        return leaf

    def get_trainable_parameters(self):
        return []


#######################################################################
#                           Constant leaves                           #
#######################################################################


class ConstantLeaf(Leaf):
    """
    A leaf that does not learn
    """

    def __init__(self, action):
        """
        Initializes the leaf

        :action: The action contained in the leaf
        """
        Leaf.__init__(self)

        self._action = action

    def get_output(self, input_):
        return self._action

    def set_reward(self, input_):
        pass

    def get_value(self):
        return 0


class ConstantLeafFactory():
    """
    A Factory for constant leaves
    """

    def create(self, params):
        action = params[0] if not isinstance(params, int) else params
        return ConstantLeaf(action)

    def get_trainable_parameters(self):
        return ["action"]
