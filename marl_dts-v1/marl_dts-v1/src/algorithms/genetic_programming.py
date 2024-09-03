#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    algorithms.genetic_programming
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Implementation of the genetic programming algorithm

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import abc
import numpy as np
from copy import deepcopy
from .common import OptMetaClass
from decisiontrees import Leaf, Condition
from operator import gt, lt, add, sub, mul
from processing_element import ProcessingElementFactory, PEFMetaClass


def safediv(a, b):
    if b == 0:
        return 0
    return a/b


class GPExpr:
    @abc.abstractmethod
    def get_output(self, input_):
        pass


class GPVar(GPExpr):
    """A variable"""
    def __init__(self, index):
        GPExpr.__init__(self)

        self._index = index

    def get_output(self, input_):
        return input_[self._index]

    def __repr__(self):
        return f"input_[{self._index}]"

    def __str__(self):
        return repr(self)


class GPArithNode(GPExpr):
    def __init__(self, op, left, right):
        GPExpr.__init__(self)

        self._op = op
        self._left = left
        self._right = right

    def get_output(self, input_):
        l = self._left.get_output(input_)
        r = self._right.get_output(input_)
        return self._op(l, r)

    def __repr__(self):
        return f"{self._op.__name__}({self._left}, {self._right})"

    def __str__(self):
        return repr(self)

    def get_left(self):
        return self._left
    
    def set_left(self, value):
        self._left = value
    
    def get_right(self):
        return self._right
    
    def set_right(self, value):
        self._right = value
    

class GPConst(GPExpr):
    def __init__(self, value):
        GPExpr.__init__(self)

        self._value = value

    def get_output(self, input_):
        return self._value

    def __repr__(self):
        return f"{self._value}"

    def __str__(self):
        return repr(self)


class GPNodeCondition:
    """
    A condition
    """

    def __init__(self, operator, left, right):
        """
        Initializes the node
        """
        self._operator = operator
        self._left = left
        self._right = right

    def get_output(self, input_):
        l = self._left.get_output(input_)
        r = self._right.get_output(input_)

        return self._operator(l, r)

    def __repr__(self):
        return f"{self._operator.__name__}({self._left}, {self._right})"

    def __str__(self):
        return repr(self)

    def get_left(self):
        return self._left
    
    def set_left(self, value):
        self._left = value

    def get_right(self):
        return self._right
    
    def set_right(self, value):
        self._right = value
    
    def get_then(self):
        return self._then
    
    def set_then(self, value):
        self._then = value
    
    def get_else(self):
        return self._else
    
    def set_else(self, value):
        self._else = value

    def empty_buffers(self):
        self._then.empty_buffers()
        self._else.empty_buffers()


class GPNodeIf(Condition):
    def __init__(self, condition, then, else_):
        self._condition = condition
        self._then = then
        self._else = else_

    def get_trainable_parameters(self):
        """
        Returns a list of parameters with their type
        (input_index, int or float) as a string.
        """
        return None

    def set_params_from_list(self, params):
        """
        Sets its parameters according to the parameters specified by
        the input list.

        :params: A list of params (int or float)
        """
        return None

    def get_output(self, input_):
        """
        Computes the output associated to its inputs (i.e. computes
        the path of the input vector (or vectors) in the tree and returns
        the decision associated to it).

        :input_: A 1D numpy array
        :returns: A 1D numpy array
        """
        if self._condition.get_output(input_):
            return self._then.get_output(input_)
        else:
            return self._else.get_output(input_)

    def empty_buffers(self):
        self._then.empty_buffers()
        self._else.empty_buffers()

    def copy(self):
        """
        Returns a copy of itself
        """
        new = deepcopy(self)
        return new

    def __repr__(self):
        return f"{self._condition}"

    def __str__(self):
        return repr(self)

    def get_then(self):
        return self._then

    def set_then(self, value):
        self._then = value

    def get_else(self):
        return self._else

    def set_else(self, value):
        self._else = value

    def get_left(self):
        return self._then

    def set_left(self, value):
        self._then = value

    def get_right(self):
        return self._else

    def set_right(self, value):
        self._else = value


class GeneticProgramming(ProcessingElementFactory, metaclass=OptMetaClass):
    """
    A class that implements the genetic programming algorithm"""

    def __init__(self, **kwargs):
        """
        Initializes the algorithm

        :pop_size: The size of the population
        :cx_prob: The crossover_probability
        :mut_prob: The mutation probability
        :tournament_size: The size of the tournament for the selection
        :l_factory: The factory for the leaves
        :bounds: dictionary containing the bounds for the two factories.
            It should contain two keys: "input_index", "float" and "action".
            The values must contain the bounds
            (a dict with keys (type, min, max))
            for all the parameters returned
            by "get_trainable_parameters"
        :max_depth: Maximum depth for the trees
        :elitist: A flag that enables elitism

        """
        self._pop_size = kwargs["pop_size"]
        self._cx_prob = kwargs["cx_prob"]
        self._mut_prob = kwargs["mut_prob"]
        self._tournament_size = kwargs["tournament_size"]
        self._l_factory = kwargs["l_factory"]
        self._bounds = kwargs["bounds"]
        self._max_depth = kwargs["max_depth"]
        self._cond_depth = kwargs.get("cond_depth", 2)
        self._elitist = kwargs.get("elitist", False)

        self._pop = self._init_pop()
        self._old_pop = None
        self._old_fit = None
        self._best = None

    def _random_var(self):
        index = np.random.randint(0, self._bounds["input_index"]["max"])
        return GPVar(index)

    def _random_const(self):
        index = np.random.uniform(self._bounds["float"]["min"], self._bounds["float"]["max"])
        return GPConst(index)

    def _random_expr(self, depth=0):
        if depth < self._cond_depth - 1:
            type_ = np.random.randint(0, 3)
        else:
            type_ = np.random.randint(0, 2)

        if type_ == 0:
            return self._random_var()
        elif type_ == 1:
            return self._random_const()
        else:
            l = self._random_expr(depth+1)
            r = self._random_expr(depth+1)
            op = np.random.choice([add, sub]) #mul, safediv
            return GPArithNode(op, l, r)

    def _random_condition(self):
        left = self._random_expr()
        right = self._random_expr()
        while isinstance(left, GPConst) and isinstance(right, GPConst):
            left = self._random_expr()
            right = self._random_expr()

        op = np.random.choice([gt, lt])

        return GPNodeIf(GPNodeCondition(op, left, right), None, None)

    def _random_leaf(self):
        tp = self._l_factory.get_trainable_parameters()

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

            if isinstance(node, Leaf) or \
               isinstance(node, GPNodeCondition) or \
               isinstance(node, GPExpr) or \
               n is None:
                continue

            if d > max_:
                max_ = d
                
            #(type(n))

            if not isinstance(n, Leaf):
                fringe.append((d+1, n._then))
                fringe.append((d+1, n._else))
        return max_

    def _reduce_expr_len(self, expr):
        fringe = [(0, expr)]

        max_ = 0
        while len(fringe) > 0:
            d, cur = fringe.pop(0)

            if isinstance(cur, GPArithNode):
                if d + 1 > self._cond_depth:
                    cur.set_left(self._random_expr(d+1))
                    cur.set_right(self._random_expr(d+1))
                else:
                    fringe.append((d+1, cur.get_left()))
                    fringe.append((d+1, cur.get_right()))
        return expr

    def _limit_cond_depth(self, root):
        """
        Limits the depth of the tree
        """
        fringe = [root]

        while len(fringe) > 0:
            cur = fringe.pop(0)

            if isinstance(cur, GPNodeIf):
                cond = cur._condition

                cond.set_left(self._reduce_expr_len(cond.get_left()))
                cond.set_right(self._reduce_expr_len(cond.get_right()))

                fringe.append(cur.get_then())
                fringe.append(cur.get_else())
        return root

    def _limit_depth(self, root):
        """
        Limits the depth of the tree
        """
        fringe = [(0, root)]

        while len(fringe) > 0:
            d, cur = fringe.pop(0)

            if isinstance(cur, GPNodeIf):
                if d + 1 == self._max_depth:
                    cur.set_then(self._random_leaf())
                    cur.set_else(self._random_leaf())
                fringe.append((d+1, cur.get_left()))
                fringe.append((d+1, cur.get_right()))
        return root

    def _init_pop(self):
        pop = []
        full = self._pop_size // 2
        grow = self._pop_size - full

        # Full
        for i in range(full):
            root = self._random_condition()
            fringe = [root]
            for d in range(self._max_depth - 2):
                # max depth - root - leaves
                new_fringe = []
                while len(fringe) > 0:
                    parent = fringe.pop(0)
                    left = self._random_condition()
                    right = self._random_condition()
                    parent.set_then(left)
                    parent.set_else(right)

                    new_fringe.append(left)
                    new_fringe.append(right)
                fringe = new_fringe

            while len(fringe) > 0:
                parent = fringe.pop(0)
                left = self._random_leaf()
                right = self._random_leaf()

                parent.set_then(left)
                parent.set_else(right)

            pop.append(root)

        # Grow
        for i in range(grow):
            root = self._get_random_leaf_or_condition()
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

                node.set_then(left)
                node.set_else(right)

                fringe.append(left)
                fringe.append(right)

            pop.append(root)

        return pop

    def ask(self):
        return self._pop[:]

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
        st1 = p1nodes[cp1][2]

        p2nodes = [(None, None, p2)]

        fringe = [p2]
        while len(fringe) > 0:
            node = fringe.pop(0)

            if not isinstance(node, Leaf) and \
               not isinstance(node, GPVar) and \
               not isinstance(node, GPConst):
                fringe.append(node.get_left())
                fringe.append(node.get_right())

                if type(node.get_left()) == type(st1):
                    p2nodes.append((node, True, node.get_left()))
                if type(node.get_right()) == type(st1):
                    p2nodes.append((node, False, node.get_right()))

        cp2 = np.random.randint(0, len(p2nodes))

        st2 = p2nodes[cp2][2]

        if cp1 != 0:
            if p1nodes[cp1][1]:
                p1nodes[cp1][0].set_then(st2)
            else:
                p1nodes[cp1][0].set_else(st2)
        else:
            p1 = st2

        if cp2 != 0:
            if p2nodes[cp2][1]:
                p2nodes[cp2][0].set_then(st1)
            else:
                p2nodes[cp2][0].set_else(st1)
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
        if not isinstance(old_node, GPNodeCondition) or \
           not isinstance(old_node, GPExpr):
            new_node = self._get_random_leaf_or_condition()
        else:
            new_node = self._random_expr()

        if not isinstance(new_node, Leaf) and \
           not isinstance(new_node, GPExpr):
            if not isinstance(old_node, Leaf):
                new_node.set_then(old_node.get_left())
                new_node.set_else(old_node.get_right())
            else:
                new_node.set_then(self._random_leaf())
                new_node.set_else(self._random_leaf())

        if p1nodes[cp1][1] is not None:
            if p1nodes[cp1][1]:
                parent.set_then(new_node)
            else:
                parent.set_else(new_node)
        else:
            p1 = new_node

        return p1

    def tell(self, fitnesses):
        
        
        new_pop = []
        
        if self._elitist and self._old_pop is not None:
            
            
            new_pop, fitnesses = self._get_elite2(self._pop, fitnesses)
            #new_pop, fitnesses = self._get_elite(self._pop, fitnesses)
            #new_pop, fitnesses = self._replace_worst(self._pop, self._old_pop, fitnesses, self._old_fit)
        
        
        selection = self._tournament_selection(fitnesses)

        n_ind = len(selection)
        
        print("new population")
        print(n_ind)
        print(len(new_pop))
        
        for i in range(0,n_ind):
            
            o1 = None
            
            p1 = self._pop[selection[i]]
            
            if np.random.uniform() < self._mut_prob:
                
                print(type(p1))
                o1 = self._mutation(p1)
            
            new_pop.append(p1 if o1 is None else o1)

        # for i in range(0, n_ind, 2):
        #     p1 = self._pop[selection[i]]

        #     if i + 1 < n_ind:
        #         p2 = self._pop[selection[i + 1]]
        #     else:
        #         p2 = None

        #     o1, o2 = None, None

        #     # Crossover
        #     if p2 is not None and np.random.uniform() < self._cx_prob:
        #         o1, o2 = self._crossover(p1, p2)

        #     # Mutation
        #     if np.random.uniform() < self._mut_prob:
        #         o1 = self._mutation(p1 if o1 is None else o1)

        #     if p2 is not None and np.random.uniform() < self._mut_prob:
        #         o2 = self._mutation(p2 if o2 is None else o2)

        #     new_pop.append(p1 if o1 is None else o1)
        #     if p2 is not None:
        #         new_pop.append(p2 if o2 is None else o2)

        for i in range(self._pop_size):
            new_pop[i] = self._limit_depth(new_pop[i])
            new_pop[i] = self._limit_cond_depth(new_pop[i])
            
        
        print("new population last")
        print(len(new_pop))

        self._old_pop = self._pop[:]
        self._old_fit = fitnesses
        self._pop = new_pop

    def _replace_worst(self, new_pop, cur_pop, new_fitnesses, old_fitnesses):
        best = np.argmax(old_fitnesses)
        if old_fitnesses[best] > np.max(new_fitnesses):
            to_replace = np.argmin(new_fitnesses)
            new_pop[to_replace] = cur_pop[best].copy()
            new_fitnesses[to_replace] = old_fitnesses[to_replace]
        return new_pop, new_fitnesses
    
    
    def _get_elite(self, cur_pop, fitnesses):
        new_pop = []
        best = np.argmax(fitnesses)
        print(best)
        new_pop.append(cur_pop[best].copy())
        
        worst = np.argmin(fitnesses)
        fitnesses = np.delete(fitnesses, worst)
        self._pop = np.delete(cur_pop, worst)
        
        return new_pop, fitnesses
    
    
    def _get_elite2(self, cur_pop, fitnesses):
        
        new_pop = []
        new_pop.append(self._best.copy())
        
        worst = np.argmin(fitnesses)
        fitnesses = np.delete(fitnesses, worst)
        self._pop = np.delete(cur_pop, worst)
        
        return new_pop, fitnesses
    
    def _record_best(self, best):
        self._best = best.copy()
        
