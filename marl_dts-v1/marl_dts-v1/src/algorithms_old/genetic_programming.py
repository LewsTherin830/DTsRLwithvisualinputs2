#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    algorithms.genetic_programming
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Implementation of the genetic programming algorithm

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import numpy as np
from decisiontreelibrary import Leaf


class GeneticProgramming:
    """
    A class that implements the genetic programming algorithm"""

    def __init__(self, pop_size, cx_prob, mut_prob, tournament_size,
                 c_factory, l_factory, bounds, max_depth):
        """
        Initializes the algorithm

        :pop_size: The size of the population
        :cx_prob: The crossover_probability
        :mut_prob: The mutation probability
        :tournament_size: The size of the tournament for the selection
        :c_factory: The factory for the conditions
        :l_factory: The factory for the leaves
        :bounds: dictionary containing the bounds for the two factories.
            It should contain two keys: "condition" and "leaf".
            The values must contain the bounds
            (a dict with keys (type, min, max))
            for all the parameters returned
            by "get_trainable_parameters"
        :max_depth: Maximum depth for the trees

        """
        self._pop_size = pop_size
        self._cx_prob = cx_prob
        self._mut_prob = mut_prob
        self._tournament_size = tournament_size
        self._c_factory = c_factory
        self._l_factory = l_factory
        self._bounds = bounds
        self._max_depth = max_depth

        self._pop = self._init_pop()
        
        
    def create_tree(self):
        
        #self._c_factory.create(params)
        #self._l_factory.create(*params)
        
        root = self._random_condition()
        fringe = [root]
        for d in range(self._max_depth - 1):
            # max depth - root - leaves
            new_fringe = []
            while len(fringe) > 0:
                parent = fringe.pop(0)
                left = self._random_condition()
                right = self._random_condition()
                parent.set_left(left)
                parent.set_right(right)

                new_fringe.append(left)
                new_fringe.append(right)
            fringe = new_fringe

        while len(fringe) > 0:
            parent = fringe.pop(0)
            left = self._random_leaf()
            right = self._random_leaf()

            parent.set_left(left)
            parent.set_right(right)

    def _random_condition(self):
        tp = self._c_factory.get_trainable_parameters()
        params = []

        #print("random_cond")
        for param in tp:
            min_ = self._bounds[param]["min"]
            max_ = self._bounds[param]["max"]
            if self._bounds[param]["type"] == "int":
                params.append(np.random.randint(min_, max_))
            elif self._bounds[param]["type"] == "float":
                params.append(np.random.uniform(min_, max_))
            else:
                raise ValueError("Unknown type")
        #print(params)
        #print(tp)
        return self._c_factory.create(params)

    def _random_leaf(self):
        tp = self._l_factory.get_trainable_parameters()
        
        #print("random_leaf")
        
        if len(tp) == 0:
            return self._l_factory.create()
        else:
            params = []

            for param in tp:
                min_ = self._bounds[param]["min"]
                max_ = self._bounds[param]["max"]
                if self._bounds[param]["type"] == "int":
                    params.append(np.random.randint(min_, max_))
                elif self._bounds[param]["type"] == "float":
                    params.append(np.random.uniform(min_, max_))
                else:
                    raise ValueError("Unknown type")
            #print(params)
            return self._l_factory.create(*params)

    def _get_random_leaf_or_condition(self):
        if np.random.uniform() < 0.5:
            return self._random_leaf()
        return self._random_condition()

    def _get_depth(self, node):
        """BFS search"""
        fringe = [(0, node)]
        max_ = 0
        while len(fringe) > 0:
            d, n = fringe.pop(0)

            if n is None:
                continue

            if d > max_:
                max_ = d

            if not isinstance(n, Leaf):
                fringe.append((d+1, n.get_left()))
                fringe.append((d+1, n.get_right()))
        return max_

    def _init_pop(self):
        pop = []
        full = self._pop_size // 2
        grow = self._pop_size - full

        # Full
        for i in range(full):
            root = self._random_condition()
            fringe = [root]
            for d in range(self._max_depth - 1):
                # max depth - root - leaves
                new_fringe = []
                while len(fringe) > 0:
                    parent = fringe.pop(0)
                    left = self._random_condition()
                    right = self._random_condition()
                    parent.set_left(left)
                    parent.set_right(right)

                    new_fringe.append(left)
                    new_fringe.append(right)
                fringe = new_fringe

            while len(fringe) > 0:
                parent = fringe.pop(0)
                left = self._random_leaf()
                right = self._random_leaf()

                parent.set_left(left)
                parent.set_right(right)

            pop.append(root)

        # Grow
        for i in range(grow):
            root = self._random_condition()
            fringe = [root]

            while len(fringe) > 0:
                node = fringe.pop(0)

                if isinstance(node, Leaf):
                    continue

                if self._get_depth(root) < self._max_depth - 1:
                    left = self._get_random_leaf_or_condition()
                    right = self._get_random_leaf_or_condition()
                else:
                    left = self._random_leaf()
                    right = self._random_leaf()

                node.set_left(left)
                node.set_right(right)

                fringe.append(left)
                fringe.append(right)

            pop.append(root)
            
        
        #print("depths:")
        #for i in pop:
            #print(self._get_depth(i))
            

        return pop

    def ask(self):
        return self._pop

    def _tournament_selection(self, fitnesses):
        n_ind = len(fitnesses)
        tournaments = np.random.choice(
            [*range(n_ind)],
            (n_ind, self._tournament_size)
        )

        selected = []

        for t in tournaments:
            max_ = float("-inf")
            argmax_ = None
            for idx in t:
                if fitnesses[idx] > max_ or argmax_ is None:
                    argmax_ = idx
                    max_ = fitnesses[idx]

            selected.append(argmax_)
        return selected

    def _crossover(self, par1, par2):
        p1, p2 = par1.copy(), par2.copy()
        cp1 = None
        cp2 = None

        p1nodes = [(None, None, p1)]

        fringe = [p1]
        while len(fringe) > 0:
            node = fringe.pop(0)

            if not isinstance(node, Leaf):
                fringe.append(node.get_left())
                fringe.append(node.get_right())

                p1nodes.append((node, True, node.get_left()))
                p1nodes.append((node, False, node.get_right()))

        cp1 = np.random.randint(0, len(p1nodes))

        p2nodes = [(None, None, p2)]

        fringe = [p2]
        while len(fringe) > 0:
            node = fringe.pop(0)

            if not isinstance(node, Leaf):
                fringe.append(node.get_left())
                fringe.append(node.get_right())

                p2nodes.append((node, True, node.get_left()))
                p2nodes.append((node, False, node.get_right()))

        cp2 = np.random.randint(0, len(p2nodes))

        st1 = p1nodes[cp1][2]
        st2 = p2nodes[cp2][2]

        if cp1 != 0:
            if p1nodes[cp1][1]:
                p1nodes[cp1][0].set_left(st2)
            else:
                p1nodes[cp1][0].set_right(st2)
        else:
            p1 = st2

        if cp2 != 0:
            if p2nodes[cp2][1]:
                p2nodes[cp2][0].set_left(st1)
            else:
                p2nodes[cp2][0].set_right(st1)
        else:
            p2 = st1
        return p1, p2

    def _mutation(self, p):
        p1 = p.copy()
        cp1 = None

        p1nodes = [(None, None, p1)]

        fringe = [p1]
        while len(fringe) > 0:
            node = fringe.pop(0)

            if not isinstance(node, Leaf):
                fringe.append(node.get_left())
                fringe.append(node.get_right())

                p1nodes.append((node, True, node.get_left()))
                p1nodes.append((node, False, node.get_right()))

        cp1 = np.random.randint(0, len(p1nodes))

        parent = p1nodes[cp1][0]
        old_node = p1nodes[cp1][2]
        new_node = self._get_random_leaf_or_condition()

        if not isinstance(new_node, Leaf):
            if not isinstance(old_node, Leaf):
                new_node.set_left(old_node.get_left())
                new_node.set_right(old_node.get_right())
            else:
                new_node.set_left(self._random_leaf())
                new_node.set_right(self._random_leaf())

        if p1nodes[cp1][1] is not None:
            if p1nodes[cp1][1]:
                parent.set_left(new_node)
            else:
                parent.set_right(new_node)
        else:
            p1 = new_node

        return p1

    def tell(self, fitnesses):
        selection = self._tournament_selection(fitnesses)

        new_pop = []
        n_ind = len(selection)

        for i in range(0, n_ind, 2):
            p1 = self._pop[selection[i]]

            if i + 1 < n_ind:
                p2 = self._pop[selection[i + 1]]
            else:
                p2 = None

            o1, o2 = None, None

            # Crossover
            if p2 is not None and np.random.uniform() < self._cx_prob:
                o1, o2 = self._crossover(p1, p2)

            # Mutation
            if np.random.uniform() < self._mut_prob:
                o1 = self._mutation(p1 if o1 is None else o1)

            if p2 is not None and np.random.uniform() < self._mut_prob:
                o2 = self._mutation(p2 if o2 is None else o2)

            new_pop.append(p1 if o1 is None else o1)
            if p2 is not None:
                new_pop.append(p2 if o2 is None else o2)
        self._pop = new_pop
