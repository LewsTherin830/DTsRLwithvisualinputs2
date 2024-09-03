#!/usr/bin/env python
import gym
import sys
sys.path.append(".")
import numpy as np
import pygmo as pg
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy.stats import norm
from functools import partial
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
dataframe = pd.DataFrame()

names = []
for i in range(1, 11):
    names += [
        (f"DE{i}", partial(pg.algorithms.de, variant=i))
    ]
    for j in range(1, 3):
        names += [
            (f"SADE{i},{j}", partial(pg.algorithms.sade, variant=i, variant_adptv=j))
        ]

for i in range(1, 3):
    names += [(f"1220DE{i}", partial(pg.algorithms.de1220, variant_adptv=i))]

names += [
    ("BC", pg.algorithms.bee_colony),
    ("CMAES", pg.algorithms.cmaes),
    ("PSO", pg.algorithms.pso),
    # pg.algorithms.gaco(gen=generations),
    # pg.algorithms.nlopt(),
    ("PSOGEN", pg.algorithms.pso_gen),
    ("EA", pg.algorithms.sea),
    ("GA", pg.algorithms.sga),
    ("XNES", pg.algorithms.xnes)
]

print(f"Testing {len(names)} algorithms")
generations_list = [*range(10, 101, 10)]
res = {name: {g: [] for g in generations_list} for name, _ in names}

for generations in generations_list:
    prob = pg.problem(cos_dist())

    for s in tqdm(range(10)):
        for i, (name, alg) in enumerate(names):
            algo = pg.algorithm(alg(gen=generations))
            pop = pg.population(prob,popsize)
            pop = algo.evolve(pop)
            res[name][generations].append(pop.champion_f)

for n in res:
    for g in res[n]:
        res[n][g] = np.mean(res[n][g])

values = [res[n][10] for n in res]
th = np.quantile(values, 0.2)

elite_res = {}
for n in res:
    if res[n][10] < th:
        elite_res[n] = res[n]

df = pd.DataFrame.from_dict(elite_res)
df.plot()
plt.show()

