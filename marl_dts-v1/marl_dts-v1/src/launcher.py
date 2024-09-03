import os
import sys
import yaml
import utils
import inspect
import logging
import argparse
import cv_modules
import algorithms
import decisiontrees
import util_processing_elements
from algorithms import continuous_optimization
from experiment_launchers.experiment import Experiment

logging.getLogger("paramiko").setLevel(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("seed", type=int)
args = parser.parse_args()

dirname = utils.get_logdir_name()
filename = args.config.split("/")[-1].replace(".yaml", "")

dirname = os.path.join("logs", filename, dirname)
os.makedirs(dirname)

config = yaml.safe_load(open(args.config))
yaml.dump(config, open(os.path.join(dirname, "config.yaml"), "w"))

exp = Experiment(args.seed, dirname, **config)
exp.run()
