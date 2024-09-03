#!/usr/bin/env python
import gym
import sys
sys.path.append(".")
import numpy as np
import pygmo as pg
import seaborn as sns
from tqdm import tqdm
from scipy.stats import norm
from concurrent import futures
from matplotlib import pyplot as plt
from cma import CMAEvolutionStrategy
from scipy.spatial.distance import cosine
from experiment_launchers.multi_agent.utils import convert_obs


ball = np.load("experiment_launchers/multi_agent/ball.npy")[:-1, :-1].flatten()
rack = np.load("experiment_launchers/multi_agent/green_rack.npy").flatten()


class cos_dist:
    def __init__(self):
        self.dim = 54

    def get_bounds(self):
        return ([0] * self.dim, [1] * self.dim)

    def get_name(self):
        return "Cosine distance"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

    def fitness(self, x):
        ball_params = x[:27]
        rack_params = x[27:]

        return cosine(ball, ball_params) + cosine(rack, rack_params),

popsize = 20
generations = 20

names = []
for i in range(1, 11):
    names += [
        (f"DE{i}", pg.algorithms.de(gen=generations, variant=i))
    ]
    for j in range(1, 3):
        names += [
            (f"SADE{i},{j}", pg.algorithms.sade(gen=generations, variant=i, variant_adptv=j))
        ]

for i in range(1, 3):
    names += [(f"1220DE{i}", pg.algorithms.de1220(gen=generations, variant_adptv=i))]

names += [
    ("BC", pg.algorithms.bee_colony(generations, generations)),
    ("CMAES", pg.algorithms.cmaes(gen=generations)),
    ("PSO", pg.algorithms.pso(gen=generations)),
    # pg.algorithms.gaco(gen=generations),
    # pg.algorithms.nlopt(),
    ("PSOGEN", pg.algorithms.pso_gen(gen=generations)),
    ("EA", pg.algorithms.sea(gen=generations)),
    ("GA", pg.algorithms.sga(gen=generations)),
    ("SA", pg.algorithms.simulated_annealing()),
    ("XNES", pg.algorithms.xnes(gen=generations))
]

print(f"Testing {len(names)} algorithms")
prob = pg.problem(cos_dist())

res = {name: [] for name, _ in names}

for s in tqdm(range(10)):
    for i, (name, alg) in enumerate(names):
        algo = pg.algorithm(alg)
        pop = pg.population(prob,popsize)
        pop = algo.evolve(pop)
        res[name].append(pop.champion_f)

values = []
for i, r in enumerate(res):
    values += [np.mean(res[r])]
th = np.quantile(values, 0.2)

x = np.linspace(0, 2*th, 400)
legends = []
for i, r in enumerate(res):
    if np.mean(res[r]) > th:
        continue
    plt.plot(x, norm.pdf(x, np.mean(res[r]), np.std(res[r])))
    legends += [r]
plt.legend(legends)
plt.show()

"""
cma = CMAEvolutionStrategy(27*2*[0.5], 0.1)
sol = cma.optimize(evaluate, iterations=100)

best = sol.result.xbest

plt.imshow(best[:27].reshape(3, 3, 3))
plt.figure()
plt.imshow(best[27:].reshape(3, 3, 3))
plt.show()
"""
