#!/usr/bin/python3
"""
Implementation of the grammatical evolution

Author: Leonardo Lucio Custode
Creation Date: 04-04-2020
Last modified: mer 6 mag 2020, 16:30:41
"""
import re
import os
import string
import numpy as np
from typing import List
from abc import abstractmethod


TAB = " " * 4


class GrammaticalEvolutionTranslator:
    def __init__(self, grammar):
        """
        Initializes a new instance of the Grammatical Evolution
        :operators: a dictionary of operators

        >>> get = GrammaticalEvolutionTranslator({"dt": ["value"]})
        >>> get.operators["dt"]
        ['value']
        >>> len(get.operators)
        1
        """
        self.operators = grammar

    def _find_candidates(self, string):
        """
        Finds all the candidates that can be replaced in the current string.
        :string: string that contains the placeholders
                (enclosed in angular brackets)

        >>> get = GrammaticalEvolutionTranslator({"dt": ["value"]})
        >>> get._find_candidates("<dt><bt><bt>")
        ['<dt>', '<bt>', '<bt>']
        >>> get._find_candidates("<dtdt><bt>")
        ['<dtdt>', '<bt>']
        """
        return re.findall("<[^> ]+>", string)

    def _find_replacement(self, candidate, gene):
        """
        Finds the value to replace to the current placeholder.

        :candidate: string containing the current placeholder.
        :gene: the value used to retrieve the value to substitute to the
                placeholder

        >>> get = GrammaticalEvolutionTranslator({"dt": ["value"]})
        >>> get._find_replacement('dt', 0)
        'value'
        >>> get._find_replacement('dt', 1)
        'value'
        >>> get._find_replacement('dt', 100)
        'value'
        >>> get = GrammaticalEvolutionTranslator({"dt": ["value1", "value2"]})
        >>> get._find_replacement('dt', 0)
        'value1'
        >>> get._find_replacement('dt', 1)
        'value2'
        >>> get._find_replacement('dt', 100)
        'value1'
        >>> get = GrammaticalEvolutionTranslator({"dt": ["value1", "value2"], 'value1': ["newval1", "newval2"]})
        >>> get._find_replacement('dt', 0)
        'value1'
        >>> get._find_replacement('dt', 1)
        'value2'
        >>> get._find_replacement('dt', 100)
        'value1'
        >>> get._find_replacement('value1', 0)
        'newval1'
        >>> get._find_replacement('value1', 1)
        'newval2'
        >>> get._find_replacement('value1', 100)
        'newval1'
        """
        key = candidate.replace("<", "").replace(">", "")
        value = self.operators[key][gene % len(self.operators[key])]
        return value

    def genotype_to_str(self, genotype):
        """
        This method translates a genotype into an executable program (python).

        :genotype: A list of genes that must be translated into a string
        :return: A tuple containing (string, number of genes used)

        >>> grammar = {
        ...     "dt": ["<even>", "<odd>", "<end>"],
        ...     "even": ['0<dt>', '2<dt>', '4<dt>', '6<dt>', '8<dt>',],
        ...     "odd": ['1<dt>', '3<dt>', '5<dt>', '7<dt>', '9<dt>',],
        ...     "end": ['']
        ... }
        >>> get = GrammaticalEvolutionTranslator(grammar)
        >>> get.genotype_to_str([0, 0, 2, 0])
        ('0', 4)
        >>> get.genotype_to_str([0, 0, 1, 0, 2, 0,])
        ('01', 6)
        """
        string = "<dt>"
        candidates = [None]
        ctr = 0
        _max_trials = 1
        genes_used = 0

        # Generate phenotype starting from the genotype
        #   If the individual runs out of genes, it restarts from the beginning, as suggested by Ryan et al 1998
        while len(candidates) > 0 and ctr <= _max_trials:
            if ctr == _max_trials and not (len(self._find_candidates(string)) == 0):
                return "", len(genotype)
            for gene in genotype:
                candidates = self._find_candidates(string)
                if len(candidates) > 0:
                    value = self._find_replacement(candidates[0], gene)
                    string = string.replace(candidates[0], value, 1)
                    genes_used += 1
                else:
                    break
            ctr += 1

        string = self._fix_indentation(string)
        return string, genes_used

    def _fix_indentation(self, string):
        """
        Adjusts the indentation by exploiting parentheses.

        :string: The string containing parentheses for each block of code
        :return: An indented string

        >>> get = GrammaticalEvolutionTranslator({"dt": ["value"]})
        >>> out = get._fix_indentation('if a<b:{if c<d:{return c;}else:{return d}}else:{return a}')
        >>> print(out)
        if a<b:
            if c<d:
                return c
        <BLANKLINE>
            else:
                return d
        <BLANKLINE>
        else:
            return a
        """
        # If parenthesis are present in the outermost block, remove them
        if string[0] == "{":
            string = string[1:-1]

        # Split in lines
        string = string.replace(";", "\n")
        string = string.replace("{", "{\n")
        string = string.replace("}", "}\n")
        lines = string.split("\n")

        fixed_lines = []
        n_tabs = 0

        # Fix lines
        for line in lines:
            if len(line) > 0:
                fixed_lines.append(TAB * n_tabs + line.replace("{", "").replace("}", ""))

                if line[-1] == "{":
                    n_tabs += 1
                while len(line) > 0 and line[-1] == "}":
                    n_tabs -= 1
                    line = line[:-1]
                if n_tabs >= 100:
                    return "None"

        return "\n".join(fixed_lines)


class Individual:
    """Represents an individual."""

    def __init__(self, genes, fitness=None, parents=None):
        """Initializes a new individual

        :genes: a list of genes
        :fitness: the fitness for the individual. Default: None.

        >>> i = Individual([0, 1, 2, 3], 5)
        >>> i._fitness
        5
        >>> i._parents
        >>> j = Individual([1, 2, 3, 4], 6)
        >>> j._fitness
        6
        >>> j._parents
        >>> k = Individual([0, 1, 2, 4], 7, [i])
        >>> k._fitness
        7
        >>> k._parents[0] is i
        True
        >>> len(k._parents)
        1
        >>> l = Individual([0, 1, 3, 4], 8, [i, j])
        >>> l._fitness
        8
        >>> l._parents[0] is i
        True
        >>> l._parents[1] is j
        True
        >>> len(l._parents)
        2
        """
        self._genes = np.array(genes)
        self._fitness = fitness
        self._parents = parents  # A list containing the indices of the parents in the previous population
        self._id = "".join(np.random.choice([*string.ascii_lowercase], 10))

    def get_genes(self):
        return self._genes

    def __repr__(self):
        return repr(self._genes).replace("array(", "").replace(")", "").replace("\n", "") + "; Fitness: {}; Parents: {}".format(self._fitness, self._parents)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return sum(self._genes != other._genes) == 0

    def copy(self):
        return Individual(self._genes[:], self._fitness, self._parents[:] if self._parents is not None else None)

    def __hash__(self):
        return hash(self._id)


class Mutator:
    """Interface for the mutation operators"""

    @abstractmethod
    def __call__(self, individual):
        pass


class UniformMutator(Mutator):
    """Uniform mutation"""

    def __init__(self, gene_probability, max_value):
        """Initializes the mutator

        :gene_probability: The probability of mutation of a single gene
        :max_value: The maximum value for a gene

        """
        Mutator.__init__(self)

        self._gene_probability = gene_probability
        self._max_value = max_value + 1

    def __call__(self, individual):
        """
        >>> ind = Individual(np.random.uniform(0, 10, 1000))
        >>> m = UniformMutator(0, 1000)
        >>> new_ind = m(ind)
        >>> sum(ind.get_genes() != new_ind.get_genes())
        0
        >>> max(new_ind.get_genes()) <= 1000
        True
        >>> m = UniformMutator(1, 1000)
        >>> new_ind = m(ind)
        >>> sum(ind.get_genes() != new_ind.get_genes())
        1000
        >>> max(new_ind.get_genes()) <= 1000
        True
        >>> m = UniformMutator(0.5, 1000)
        >>> new_ind = m(ind)
        >>> 600 > sum(ind.get_genes() != new_ind.get_genes()) > 400
        True
        >>> max(new_ind.get_genes()) <= 1000
        True
        """
        mutated_genes = np.random.uniform(0, 1, len(individual._genes)) < self._gene_probability
        gene_values = np.random.randint(0, self._max_value, sum(mutated_genes))
        genes = individual._genes.copy()
        genes[mutated_genes] = gene_values
        new_individual = Individual(genes, parents=individual._parents)
        return new_individual

    def __repr__(self):
        return "UniformMutator({}, {})".format(self._gene_probability, self._max_value)


class Crossover:
    """Interface for the crossover operators"""

    @abstractmethod
    def __call__(self, individual1, individual2) -> List:
        pass


class OnePointCrossover(Crossover):
    """One point crossover"""

    def __call__(self, individual1, individual2):
        """
        >>> ind1 = Individual(np.random.uniform(0, 10, 1000))
        >>> ind2 = Individual(np.random.uniform(0, 10, 1000))
        >>> xover = OnePointCrossover()
        >>> ni1, ni2 = xover(ind1, ind2)
        >>> splitting_point = np.where(ni1.get_genes() != ind1.get_genes())[0][0]
        >>> sum([ng != g for g, ng in zip(ind1.get_genes()[:splitting_point], ni1.get_genes()[:splitting_point])])
        0
        >>> sum([ng != g for g, ng in zip(ind2.get_genes()[:splitting_point], ni2.get_genes()[:splitting_point])])
        0
        >>> sum([ng != g for g, ng in zip(ind2.get_genes()[splitting_point:], ni1.get_genes()[splitting_point:])])
        0
        >>> sum([ng != g for g, ng in zip(ind1.get_genes()[splitting_point:], ni2.get_genes()[splitting_point:])])
        0
        """
        point = np.random.randint(0, len(individual1._genes) - 2)

        new_individuals = [Individual([*individual1._genes[:point], *individual2._genes[point:]])]
        new_individuals.append(Individual([*individual2._genes[:point], *individual1._genes[point:]]))
        return new_individuals

    def __repr__(self):
        return "OnePointCrossover"


class Selection:
    """Abstract class for the selection operators"""

    def __init__(self, logfile=None):
        self._logfile = logfile

    @abstractmethod
    def __call__(self, fitnesses: List) -> List:
        """ Returns a sorted list of indices, so that one can simply crossover the adjacent individuals """
        pass

    def _log(self, msg):
        if self._logfile is not None:
            with open(self._logfile, "a") as f:
                f.write(msg + "\n")


class BestSelection(Selection):
    """Interface for the selection operators"""

    @abstractmethod
    def __call__(self, fitnesses: List) -> List:
        """ Returns a sorted list of indices, so that one can simply crossover the adjacent individuals """
        order = sorted(range(len(fitnesses)), key=lambda x: fitnesses[x], reverse=True)
        self._log("The individuals (Fitnesses {}) have been sorted as {}.".format(fitnesses, order))
        return order

    def __repr__(self):
        return "BestSelection"


class TournamentSelection(Selection):
    """Tournament selection"""

    def __init__(self, tournament_size, logfile=None):
        """Initializes a new tournament selection

        :tournament_size: number of individual in the tournament
        """
        Selection.__init__(self, logfile)

        self._tournament_size = tournament_size

    def __call__(self, fitnesses):
        """
        >>> ts = TournamentSelection(4)
        >>> n_bests = [sum(np.array(ts([1, 2, 3, 4])) == 3) for i in range(100)]
        >>> np.mean(n_bests) == 4
        True
        >>> ts = TournamentSelection(3)
        >>> n_bests = [sum(np.array(ts([1, 2, 3, 4])) == 3) for i in range(100)]
        >>> np.mean(n_bests) > 2.5
        True
        """
        # tournaments = np.random.randint(0, len(fitnesses), (len(fitnesses), self._tournament_size))
        tournaments = np.array([np.random.choice(list(range(len(fitnesses))), size=(self._tournament_size), replace=False) for _ in range(len(fitnesses))])
        selection = []

        for i in tournaments:
            # print(i)
            selection.append(i[np.argmax([fitnesses[j] for j in i])])
            # print(selection[-1])
            self._log("Individuals in the tournament: {} (fitnesses: {}), selected: {}".format(list(i), [fitnesses[j] for j in i], selection[-1]))

        return selection

    def __repr__(self):
        return "TournamentSelection({})".format(self._tournament_size)


class Replacement:
    """ Interface for the replacement operators """

    @abstractmethod
    def __call__(self, old_pop: List, new_pop: List) -> List:
        pass


class NoReplacement(Replacement):
    """Uses the new population."""

    def __call__(self, old_pop, new_pop):
        return new_pop


class ReplaceIfBetter(Replacement):
    """Replaces the parents if the new individual is better"""

    def __init__(self, logfile=None):
        self._logfile = logfile

    def _log(self, msg):
        if self._logfile is not None:
            with open(self._logfile, "a") as f:
                f.write(msg + "\n")

    def __call__(self, old_pop, new_pop):
        """
        >>> old_pop = [Individual(np.random.uniform(0, 1000, 1000), 10) for _ in range(10)]
        >>> new_pop = [Individual(np.random.uniform(0, 1000, 1000), 9, [old_pop[i]]) for i in range(10)]
        >>> r = ReplaceIfBetter()
        >>> replaced = r(old_pop, new_pop)
        >>> sum([sum(rep.get_genes() == old.get_genes()) == len(rep.get_genes()) for rep, old in zip(replaced, old_pop)])
        10
        >>> sum([sum(rep.get_genes() == new.get_genes()) == len(rep.get_genes()) for rep, new in zip(replaced, new_pop)])
        0
        >>> old_pop = [Individual(np.random.uniform(0, 1000, 1000), 10) for _ in range(10)]
        >>> new_pop = [Individual(np.random.uniform(0, 1000, 1000), 11, [old_pop[i]]) for i in range(10)]
        >>> r = ReplaceIfBetter()
        >>> replaced = r(old_pop, new_pop)
        >>> sum([sum(rep.get_genes() == old.get_genes()) == len(rep.get_genes()) for rep, old in zip(replaced, old_pop)])
        0
        >>> sum([sum(rep.get_genes() == new.get_genes()) == len(rep.get_genes()) for rep, new in zip(replaced, new_pop)])
        10
        >>> old_pop = [Individual(np.random.uniform(0, 1000, 1000), 10) for _ in range(10)]
        >>> new_pop = [Individual(np.random.uniform(0, 1000, 1000), 10 + (1 if (i % 2 == 0) else -1), [old_pop[i]]) for i in range(10)]
        >>> r = ReplaceIfBetter()
        >>> replaced = r(old_pop, new_pop)
        >>> ([sum(rep.get_genes() == old.get_genes()) == len(rep.get_genes()) for rep, old in zip(replaced, old_pop)])
        [False, True, False, True, False, True, False, True, False, True]
        >>> ([sum(rep.get_genes() == new.get_genes()) == len(rep.get_genes()) for rep, new in zip(replaced, new_pop)])
        [True, False, True, False, True, False, True, False, True, False]
        >>> old_pop = [Individual(np.random.uniform(0, 1000, 1000), 10) for _ in range(10)]
        >>> new_pop = [Individual(np.random.uniform(0, 1000, 1000), 10, [old_pop[i]]) for i in range(10)]
        >>> r = ReplaceIfBetter()
        >>> replaced = r(old_pop, new_pop)
        >>> sum([sum(rep.get_genes() == old.get_genes()) == len(rep.get_genes()) for rep, old in zip(replaced, old_pop)])
        10
        >>> sum([sum(rep.get_genes() == new.get_genes()) == len(rep.get_genes()) for rep, new in zip(replaced, new_pop)])
        0
        """
        final_population = {old: old.copy() for old in old_pop}
        assert len(final_population) == len(old_pop), f"Initially {len(final_population)} != {len(old_pop)}"
        assert len(final_population) == len(new_pop), f"Initially {len(final_population)} != {len(new_pop)}"

        for ind in new_pop:
            # Here I use final_population instead of directly the parents because
            # They may have already been replaced by someone else
            if ind._parents is None:
                continue
            parents_fitnesses = {p: final_population[p]._fitness for p in ind._parents}

            worst_fitness = float("inf")
            worst_parent = None

            for parent, fitness in parents_fitnesses.items():
                if fitness < worst_fitness:
                    worst_fitness = fitness
                    worst_parent = parent

            if ind._fitness > worst_fitness:
                final_population[worst_parent] = ind

        final_population = list(final_population.values())
        assert len(final_population) == len(old_pop), f"{len(final_population)} != {len(old_pop)}"
        assert len(final_population) == len(new_pop), f"{len(final_population)} != {len(new_pop)}"
        return final_population

    def __repr__(self):
        return "ReplaceIfBetter"


class Replacement:
    """ Interface for the replacement operators """

    @abstractmethod
    def __call__(self, old_pop: List, new_pop: List) -> List:
        pass


class ReplaceWithOldIfWorse(Replacement):
    """Replaces the parents if the new individual is better"""

    def __init__(self, logfile=None):
        self._logfile = logfile

    def _log(self, msg):
        if self._logfile is not None:
            with open(self._logfile, "a") as f:
                f.write(msg + "\n")

    def __call__(self, old_pop, new_pop):
        """
        >>> old_pop = [Individual(np.random.uniform(0, 1000, 1000), 10) for _ in range(10)]
        >>> new_pop = [Individual(np.random.uniform(0, 1000, 1000), 9, [old_pop[i]]) for i in range(10)]
        >>> r = ReplaceWithOldIfWorse()
        >>> replaced = r(old_pop, new_pop)
        >>> sum([sum(rep.get_genes() == old.get_genes()) == len(rep.get_genes()) for rep, old in zip(replaced, old_pop)])
        10
        >>> sum([sum(rep.get_genes() == new.get_genes()) == len(rep.get_genes()) for rep, new in zip(replaced, new_pop)])
        0
        >>> old_pop = [Individual(np.random.uniform(0, 1000, 1000), 10) for _ in range(10)]
        >>> new_pop = [Individual(np.random.uniform(0, 1000, 1000), 11, [old_pop[i]]) for i in range(10)]
        >>> r = ReplaceWithOldIfWorse()
        >>> replaced = r(old_pop, new_pop)
        >>> sum([sum(rep.get_genes() == old.get_genes()) == len(rep.get_genes()) for rep, old in zip(replaced, old_pop)])
        0
        >>> sum([sum(rep.get_genes() == new.get_genes()) == len(rep.get_genes()) for rep, new in zip(replaced, new_pop)])
        10
        >>> old_pop = [Individual(np.random.uniform(0, 1000, 1000), 10) for _ in range(10)]
        >>> new_pop = [Individual(np.random.uniform(0, 1000, 1000), 10 + (1 if (i % 2 == 0) else -1), [old_pop[i]]) for i in range(10)]
        >>> r = ReplaceWithOldIfWorse()
        >>> replaced = r(old_pop, new_pop)
        >>> ([sum(rep.get_genes() == old.get_genes()) == len(rep.get_genes()) for rep, old in zip(replaced, old_pop)])
        [False, True, False, True, False, True, False, True, False, True]
        >>> ([sum(rep.get_genes() == new.get_genes()) == len(rep.get_genes()) for rep, new in zip(replaced, new_pop)])
        [True, False, True, False, True, False, True, False, True, False]
        >>> old_pop = [Individual(np.random.uniform(0, 1000, 1000), 10) for _ in range(10)]
        >>> new_pop = [Individual(np.random.uniform(0, 1000, 1000), 10, [old_pop[i]]) for i in range(10)]
        >>> r = ReplaceWithOldIfWorse()
        >>> replaced = r(old_pop, new_pop)
        >>> sum([sum(rep.get_genes() == old.get_genes()) == len(rep.get_genes()) for rep, old in zip(replaced, old_pop)])
        0
        >>> sum([sum(rep.get_genes() == new.get_genes()) == len(rep.get_genes()) for rep, new in zip(replaced, new_pop)])
        10
        """
        final_pop = []

        for o in old_pop:
            o._parents = None

        for i in np.arange(0, len(new_pop), step=2):
            i1, i2 = new_pop[i:i + 2]
            if i1._parents is None:
                # First generation
                assert i2._parents is None
                final_pop.append(i1.copy())
                final_pop.append(i2.copy())
                continue

            if len(i1._parents) == 1:
                for ind in [i1, i2]:
                    parent = ind._parents[0]
                    if ind._fitness >= parent._fitness:
                        final_pop.append(ind.copy())
                    else:
                        final_pop.append(parent.copy())
            else:
                p1, p2 = [k for k in i1._parents]
                fn1, fn2 = [k._fitness for k in [i1, i2]]
                fo1, fo2 = [k._fitness for k in [p1, p2]]

                newbest = max(fn1, fn2)
                newworst = min(fn1, fn2)
                oldbest = max(fo1, fo2)
                oldworst = min(fo1, fo2)

                if (newworst > oldbest) or (newbest > oldbest and newworst > oldworst):
                    final_pop.append(i1.copy())
                    final_pop.append(i2.copy())
                else:
                    if newbest < oldworst:
                        final_pop.append(p1.copy())
                        final_pop.append(p2.copy())
                    else:
                        if fn1 > fn2:
                            final_pop.append(i1.copy())
                        else:
                            final_pop.append(i2.copy())

                        if fo1 == oldbest:
                            final_pop.append(p1.copy())
                        else:
                            final_pop.append(p2.copy())

        assert len(final_pop) == len(old_pop) == len(new_pop)
        return final_pop

    def __repr__(self):
        return "ReplaceWithOldIfWorse"


class GrammaticalEvolution:
    """A class that implements grammatical evolution (Ryan et al. 1995)"""

    def __init__(self, **kwargs):
        """
        Initializes the optimizer

        :pop_size: the size of the population
        :mutation: the mutation operator
        :crossover: the crossover operator
        :selection: the selection operator
        :replacement: the replacement operator
        :mut_prob: the mutation probability
        :cx_prob: the crossover probability
        :genotype_length: the length of the genotype
        :max_int: the biggest constant that can be contained in the genotype (random number ins the range [0, max_int] are generated)

        """
        self._pop_size = kwargs["pop_size"]
        self._mutation = kwargs["mutation"]
        self._crossover = kwargs["crossover"]
        self._selection = kwargs["selection"]
        self._replacement = kwargs["replacement"]
        self._mut_prob = kwargs["mut_prob"]
        self._cx_prob = kwargs["cx_prob"]
        self._genotype_length = kwargs["genotype_length"]
        self._max_int = kwargs.get("max_int", 10000)
        logdir = None

        self._individuals = []
        self._logfile = os.path.join(logdir, "grammatical_evolution.log") if logdir is not None else None
        self._init_pop()
        self._old_individuals = []
        self._updated = False  # To detect the first generation

    def _init_pop(self):
        """Initializes the population"""
        for i in range(self._pop_size):
            self._individuals.append(self._random_individual())
            self._log("INIT", "Individual {}:\n{}".format(i, self._individuals[-1]))

    def _log(self, tag, string):
        if self._logfile is not None:
            with open(self._logfile, "a") as f:
                f.write("[{}] {}\n".format(tag, string))

    def _random_individual(self):
        """ Generates a random individual """
        return Individual(np.random.randint(0, self._max_int + 1, self._genotype_length))

    def ask(self):
        """
        Returns the current population
        :pop_size: the size of the population
        :mutation: the mutation operator
        :crossover: the crossover operator
        :selection: the selection operator
        :replacement: the replacement operator
        :mut_prob: the mutation probability
        :cx_prob: the crossover probability
        :genotype_length: the length of the genotype
        :max_int: the biggest constant that can be contained in the genotype (so random number in the range [0, max_int] are generated)

        >>> occ = []
        >>> for i in range(100):
        ...     ge = GrammaticalEvolution(
        ...         pop_size=10, mutation=UniformMutator(0, 1000), crossover=OnePointCrossover(),
        ...         selection=TournamentSelection(10), replacement=ReplaceWithOldIfWorse(),
        ...         mut_prob=1, cx_prob=0, genotype_length=100, max_int=1000, logdir=None
        ...     )
        ...     pop = ge.ask()
        ...     ge.tell([1] + [0] * (len(pop) - 1))
        ...     new_pop = ge.ask()
        ...     occ.append(sum([np.linalg.norm(ni._parents[0].get_genes() - pop[0].get_genes()) == 0 for ni in new_pop]))
        >>> np.mean(occ) == 10
        True
        >>> occ = []
        >>> for i in range(100):
        ...     ge = GrammaticalEvolution(
        ...         pop_size=10, mutation=UniformMutator(0, 1000), crossover=OnePointCrossover(),
        ...         selection=TournamentSelection(2), replacement=ReplaceWithOldIfWorse(),
        ...         mut_prob=1, cx_prob=0, genotype_length=100, max_int=1000, logdir=None
        ...     )
        ...     pop = ge.ask()
        ...     ge.tell([1] + [0] * (len(pop) - 1))
        ...     new_pop = ge.ask()
        ...     tmp = [sum(ni._parents[0].get_genes() == pop[0].get_genes()) == 100 for ni in new_pop]
        ...     occ.append(sum(tmp))
        >>> 1.5 < np.mean(occ) < 2.5
        True
        >>> occ = []
        >>> for i in range(100):
        ...     ge = GrammaticalEvolution(
        ...         pop_size=10, mutation=UniformMutator(0, 1000), crossover=OnePointCrossover(),
        ...         selection=TournamentSelection(3), replacement=ReplaceWithOldIfWorse(),
        ...         mut_prob=1, cx_prob=0, genotype_length=100, max_int=1000, logdir=None
        ...     )
        ...     pop = ge.ask()
        ...     ge.tell([1] + [0] * (len(pop) - 1))
        ...     new_pop = ge.ask()
        ...     tmp = [sum(ni._parents[0].get_genes() == pop[0].get_genes()) == 100 for ni in new_pop]
        ...     occ.append(sum(tmp))
        >>> 2.8 < np.mean(occ) < 3.8
        True
        >>> a1 = []
        >>> a2 = []
        >>> for i in range(100):
        ...     ge = GrammaticalEvolution(
        ...         pop_size=10, mutation=UniformMutator(0, 1000), crossover=OnePointCrossover(),
        ...         selection=TournamentSelection(10), replacement=ReplaceWithOldIfWorse(),
        ...         mut_prob=0, cx_prob=0, genotype_length=100, max_int=1000, logdir=None
        ...     )
        ...     pop = ge.ask()
        ...     ge.tell([1] + [0] * (len(pop) - 1))
        ...     new_pop = ge.ask()
        ...     tmp = [sum(ni.get_genes() == pop[0].get_genes()) == 100 for ni in new_pop]
        ...     a1.append(sum(tmp))
        ...     tmp = [ni.get_genes() is pop[0].get_genes() for ni in new_pop]
        ...     a2.append(sum(tmp))
        ...     old_val = pop[0].get_genes()[20]
        ...     new_pop[0]._genes[20] = old_val + 1
        ...     for ni in new_pop[1:]:
        ...         if ni.get_genes()[20] != old_val:
        ...             print('New individuals share the genes')
        ...     if pop[0].get_genes()[20] != old_val:
        ...         print('Old individuals and new ones share the genes')
        >>> np.mean(a1) == 10
        True
        >>> np.mean(a2) == 0
        True
        >>> a1 = []
        >>> a2 = []
        >>> for i in range(100):
        ...     ge = GrammaticalEvolution(
        ...         pop_size=10, mutation=UniformMutator(0, 1000), crossover=OnePointCrossover(),
        ...         selection=TournamentSelection(10), replacement=ReplaceWithOldIfWorse(),
        ...         mut_prob=0, cx_prob=0, genotype_length=100, max_int=1000, logdir=None
        ...     )
        ...     pop = ge.ask()
        ...     ge.tell([0] * (len(pop) // 2) + [1] + [0] * (len(pop) - len(pop) //2 - 1))
        ...     new_pop = ge.ask()
        ...     ind_idx = len(pop) // 2
        ...     tmp = [sum(ni.get_genes() == pop[ind_idx].get_genes()) == 100 for ni in new_pop]
        ...     a1.append(sum(tmp))
        ...     tmp = [ni.get_genes() is pop[ind_idx].get_genes() for ni in new_pop]
        ...     a2.append(sum(tmp))
        ...     old_val = pop[ind_idx].get_genes()[20]
        ...     new_pop[0]._genes[20] = old_val + 1
        ...     for ni in new_pop[1:]:
        ...         if ni.get_genes()[20] != old_val:
        ...             print('New individuals share the genes')
        ...     if pop[ind_idx].get_genes()[20] != old_val:
        ...         print('Old individuals and new ones share the genes')
        >>> np.mean(a1) == 10
        True
        >>> np.mean(a2) == 0
        True
        >>> a1 = []
        >>> a2 = []
        >>> for i in range(100):
        ...     ge = GrammaticalEvolution(
        ...         pop_size=10, mutation=UniformMutator(0, 1000), crossover=OnePointCrossover(),
        ...         selection=TournamentSelection(10), replacement=ReplaceWithOldIfWorse(),
        ...         mut_prob=1, cx_prob=1, genotype_length=100, max_int=1000, logdir=None
        ...     )
        ...     pop = ge.ask()
        ...     ge.tell([0] * (len(pop) // 2) + [1] + [0] * (len(pop) - len(pop) //2 - 1))
        ...     new_pop = ge.ask()
        ...     ind_idx = len(pop) // 2
        ...     tmp = [sum(ni.get_genes() == pop[ind_idx].get_genes()) == 100 for ni in new_pop]
        ...     a1.append(sum(tmp))
        ...     tmp = [ni.get_genes() is pop[ind_idx].get_genes() for ni in new_pop]
        ...     a2.append(sum(tmp))
        ...     old_val = pop[ind_idx].get_genes()[0]
        ...     new_pop[0]._genes[0] = old_val + 1
        ...     for ni in new_pop[1:]:
        ...         if ni.get_genes()[0] != old_val:
        ...             print([x.get_genes()[0] for x in [ni, *ni._parents]], old_val)
        ...             print('New individuals share the genes')
        ...     if pop[ind_idx].get_genes()[0] != old_val:
        ...         print('Old individuals and new ones share the genes')
        >>> np.mean(a1) == 10
        True
        >>> np.mean(a2) == 0
        True
        >>> for i in range(10):
        ...     ge = GrammaticalEvolution(
        ...         pop_size=10, mutation=UniformMutator(0.1, 1), crossover=OnePointCrossover(),
        ...         selection=TournamentSelection(2), replacement=ReplaceWithOldIfWorse(),
        ...         mut_prob=1, cx_prob=1, genotype_length=10, max_int=1, logdir=None
        ...     )
        ...     for gen in range(100):
        ...         pop = ge.ask()
        ...         ge.tell([sum(p.get_genes()) for p in pop]) 
        ...     if np.mean([sum(p.get_genes()) for p in pop]) <= 9:
        ...         print('Optimization not working')
        """
        self._old_individuals = self._individuals[:]
        if self._updated:
            self._individuals = []
            _sorted_pop = [self._old_individuals[j].copy() for j in self._selection([i._fitness for i in self._old_individuals])]

            for s in _sorted_pop:
                s._parents = None

            for s in _sorted_pop:
                s._parents = None

            self._log("POPULATION", "Sorted population:\n" + "\n".join("Individual {}:\n{}".format(srt_idx, _sorted_pop[srt_idx]) for srt_idx in range(len(_sorted_pop))))

            cx_random_numbers = np.random.uniform(0, 1, len(_sorted_pop)//2)
            m_random_numbers = np.random.uniform(0, 1, len(_sorted_pop))

            # Crossover
            for index, cxp in enumerate(cx_random_numbers):
                ind1, ind2 = _sorted_pop[2 * index: 2 * index + 2]
                if cxp < self._cx_prob:
                    self._log("CX", "cx happened between individual {} and {}".format(2 * index, 2 * index + 1))
                    self._individuals.extend(self._crossover(ind1, ind2))
                    self._individuals[-1]._parents = [ind1, ind2]
                    self._individuals[-2]._parents = [ind1, ind2]
                    self._log("CX", "Individual {} has parents [{}, {}] (Fitness [{}, {}])".format(2 * index, 2 * index, 2 * index + 1, ind1._fitness, ind2._fitness))
                    self._log("CX", "Individual {} has parents [{}, {}] (Fitness [{}, {}])".format(2 * index + 1, 2 * index, 2 * index + 1, ind1._fitness, ind2._fitness))
                else:
                    self._log("CX", "cx did not happen between individual {} and {}".format(2 * index, 2 * index + 1))
                    self._individuals.extend([Individual(ind1._genes), Individual(ind2._genes)])
                    self._individuals[-1]._parents = [ind2]
                    self._individuals[-2]._parents = [ind1]
                    self._log("CX", "Individual {} has parents [{}] (Fitness {})".format(2 * index, 2 * index, ind1._fitness))
                    self._log("CX", "Individual {} has parents [{}] (Fitness {})".format(2 * index + 1, 2 * index + 1, ind2._fitness))

            if len(_sorted_pop) % 2 == 1:
                self._individuals.append(_sorted_pop[-1])
                self._individuals[-1]._parents = [_sorted_pop[-1]]

            # Mutation
            for i, mp in enumerate(m_random_numbers):
                if mp < self._mut_prob:
                    self._log("MUT", "Mutation occurred for individual {}".format(i))
                    self._individuals[i] = self._mutation(self._individuals[i])
            self._old_individuals = _sorted_pop
        return self._individuals

    def tell(self, fitnesses):
        """
        Assigns the fitness for each individual

        :fitnesses: a list of numbers (the higher the better) associated (by index) to the individuals
        """
        for index, (i, f) in enumerate(zip(self._individuals, fitnesses)):
            if i._parents is not None:
                self._log("FITNESS", "Individual {} has fitness {}. Its parents ({}) have fitnesses {}".format(index, f, i._parents, [k._fitness for k in i._parents]))
            else:
                self._log("FITNESS", "Individual {} has fitness {}".format(index, f))
            i._fitness = f
        self._update_population()

    def _update_population(self):
        """ Creates the next population """
        self._updated = True
        self._individuals = self._replacement(self._old_individuals, self._individuals)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
