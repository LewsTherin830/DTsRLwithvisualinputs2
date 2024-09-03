#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    marl_dts.test_algorithms
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Tests for the algorithms module

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import pytest
import numpy as np
# Selection operators
from ..src.algorithms.grammatical_evolution import BestSelection, \
        TournamentSelection
# Crossover operators
from ..src.algorithms.grammatical_evolution import OnePointCrossover
# Mutation operators
from ..src.algorithms.grammatical_evolution import UniformMutator
# Replacement operators
from ..src.algorithms.grammatical_evolution import ReplaceIfBetter, \
        ReplaceWithOldIfWorse, NoReplacement
# Aux stuff
from ..src.algorithms.grammatical_evolution import Individual
# Algorithms
from ..src.algorithms.grammatical_evolution import GrammaticalEvolution, \
        GrammaticalEvolutionTranslator


#######################################################################
#                              Operators                              #
#######################################################################

#########################
#  Selection operators  #
#########################

n_ind = 10
n_runs = 10
run_ids = [*range(n_runs)]
tournament_sizes = [*range(1, n_ind)]
n_iterations = 100

ts_inputs = []

for r in run_ids:
    for ts in tournament_sizes:
        ts_inputs.append((ts, r))


def test_gebs0():
    fitnesses = [*range(n_ind)]

    selected_individuals = BestSelection()(fitnesses)

    for i in range(n_ind):
        assert selected_individuals[i] == fitnesses[-(i+1)]


@pytest.mark.parametrize("run_id", run_ids)
def test_gebs1(run_id):
    np.random.seed(run_id)

    fitnesses = [*range(n_ind)]

    perm_fitnesses = np.random.permutation(fitnesses)
    selected_individuals = BestSelection()(perm_fitnesses)

    for i in range(n_ind):
        assert perm_fitnesses[selected_individuals[i]] == n_ind - i - 1


@pytest.mark.parametrize("run_id", run_ids)
def test_gets0(run_id):
    np.random.seed(run_id)
    fitnesses = [*range(n_ind)]

    selected_individuals = TournamentSelection(n_ind * 100)(fitnesses)
    # Probability of not getting the most fit individual is quite low
    print(selected_individuals)

    cnt = 0
    for i in selected_individuals:
        if i != n_ind - 1:
            cnt += 1
    assert cnt < 2  # It may happen in a case


@pytest.mark.parametrize("tournament_size, run_id", ts_inputs)
def test_gets1(tournament_size, run_id):
    np.random.seed(run_id)
    fitnesses = [*range(n_ind)]

    selected_individuals = []
    for i in range(n_iterations):
        selected_individuals.extend(
            TournamentSelection(tournament_size)(fitnesses)
        )
    # Probability of not getting the most fit individual is:
    #    ((n_ind - 1)/n_ind) ^ tournament_size

    count = 0
    for i in selected_individuals:
        if i == n_ind - 1:
            count += 1
        # Probability of getting 0 is 1/(n_ind^tournament_size)
        if 1/(n_ind ** tournament_size) < 0.000001:
            assert i != 0
    # Give it some tolerance
    assert count >= n_iterations * np.floor((
        n_ind - ((1 - 1/n_ind) ** tournament_size) * n_ind
        ) * 0.9)
