#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    src.mapper
    ~~~~~~~~~~

    This module implement several classes which perform mapping of jobs

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import os
import utils
import shutil
import pickle
import numpy as np
from copy import deepcopy
from scp import SCPClient
from time import time, sleep
from paramiko import SSHClient
from abc import abstractmethod
from tasks import TaskMetaClass
from sklearn.cluster import DBSCAN
from joblib import Parallel, delayed


class MapperMetaClass(type):
    _registry = {}

    def __init__(cls, clsname, bases, methods):
        super().__init__(clsname, bases, methods)
        MapperMetaClass._registry[cls.__name__] = cls

    @staticmethod
    def get(class_name):
        """
        Retrieves the class associated to the string

        :class_name: The name of the class
        :returns: A class
        """
        return MapperMetaClass._registry[class_name]


class Mapper:
    """Base class"""

    @abstractmethod
    def map(self, function, inputs, config):
        """
        Performs mapping

        :function: The function to map
        :inputs: The list of inputs to pass to the function
        :config: The dictionary containing the parameters
        :returns: A list of outputs returned by the function for each input
        """
        pass


class SingleProcessMapper(Mapper, metaclass=MapperMetaClass):
    """
    A mapper which executes the jobs on a single process
    """

    def map(self, function, inputs, config):
        outputs = []

        for x in inputs:
            outputs.append(function(x))
        return outputs


class MultiProcessMapper(Mapper, metaclass=MapperMetaClass):
    """
    A mapper which executes the jobs on multiple processes on a single machine
    """

    def __init__(self, **kwargs):
        """
        Initializes the mapper

        - n_jobs: the number of processes to use
        """
        Mapper.__init__(self)
        self._n_jobs = kwargs["n_jobs"]

    def map(self, function, inputs, config):
        return Parallel(self._n_jobs)\
                (delayed(function)(x, config) for x in inputs)


class MultiMachineMapper(Mapper, metaclass=MapperMetaClass):
    """
    A mapper that parallelizes across machines
    """

    def __init__(self, machines):
        """
        Initalizes the mapper

        - machines:
            - hostname
            - username
            - command: command used to launch the job
            - max_jobs: the maximum number of jobs that the machine\
                        allows
            - maximum_time: the maximum time allowed for the job (s)
            - kill_cmd: the command to kill the current runs
        """
        Mapper.__init__(self)
        self._machines = machines

        self._dirname = self.__class__.__name__
        try:
            # Delete old files
            shutil.rmtree(self._dirname)
        except FileNotFoundError:
            pass
        # Create new dir
        os.makedirs(self._dirname)

    def _get_connections(self):
        # Init connections
        connections = []

        for m in self._machines:
            connections.append(self._get_connection(m))
        return connections

    def _get_connection(self, config):
        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.connect(config["hostname"], username=config["username"])

        return ssh

    def _create_pkl_reuse(self, obj, fname):
        files = os.listdir(self._dirname)

        if fname not in files:
            self._create_pkl(obj, fname)

    def _read_pkl(self, fname):
        return pickle.load(open(os.path.join(self._dirname, fname), "rb"))

    def _create_pkl(self, obj, fname):
        pickle.dump(obj, open(os.path.join(self._dirname, fname), "wb"))

    def _send_pkl_reuse(self, ssh, fname):
        _, o, _ = ssh.exec_command(f"ls")
        files = str(o.read())

        if fname not in files:
            self._send_pkl(ssh, fname)

    def _send_pkl(self, ssh, fname):
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(os.path.join(self._dirname, fname), fname)

    def _retrieve_pkl(self, ssh, fname):
        with SCPClient(ssh.get_transport()) as scp:
            scp.get(fname, os.path.join(self._dirname, fname))

    def _start_job(self, conn, machine, fname, chunk_name, cname):
        _, _, stderr = conn.exec_command(
            f"{machine['command']} {fname} {chunk_name} {cname}"
        )
        stderr = stderr.read()
        return stderr

    def _get_chunk_name(self, i):
        return f"{i}.pkl"

    def _get_output_name(self, chunk):
        return f"output_{chunk}"

    def _get_chunk_id(self, chunk):
        return int(chunk.split(".")[0])

    def map(self, function, inputs, config):
        # Upload function and config
        fname = f"{function.__name__}.pkl"
        cname = "config.pkl"
        self._create_pkl_reuse(function, fname)
        self._create_pkl_reuse(config, cname)

        for c in self._get_connections():
            # Avoid re-uploading the function pickle each time
            self._send_pkl_reuse(c, fname)
            self._send_pkl_reuse(c, cname)
            c.close()

        # Upload individuals
        total_chunks = 0

        for m in self._machines:
            total_chunks += m["max_jobs"]

        # For the moment, split them uniformly across the machines
        chunk_size = int(np.ceil(len(inputs) / total_chunks))

        for i in range(total_chunks):
            chunk_name = self._get_chunk_name(i)
            chunk = inputs[i*chunk_size:(i+1)*chunk_size]
            self._create_pkl(chunk, chunk_name)

        # Upload and launch
        cur_chunk = 0
        association = {}
        launch_time = {}
        for machine, conn in zip(self._machines, self._get_connections()):
            for job_i in range(machine["max_jobs"]):
                chunk_name = self._get_chunk_name(cur_chunk)
                self._send_pkl(conn, chunk_name)

                if machine["hostname"] not in association:
                    association[machine["hostname"]] = []
                    launch_time[machine["hostname"]] = []
                association[machine["hostname"]].append(chunk_name)

                stderr = self._start_job(conn, machine, fname, chunk_name, cname)

                launch_time[machine["hostname"]].append(time())
                if len(stderr) > 0:
                    print(f"Error during execution on {m['hostname']}: {stderr}")
                cur_chunk += 1
            conn.close()

        # Wait for the completion
        outputs = [None] * total_chunks
        completed = [False] * len(self._machines)
        connections = self._get_connections()
        while not all(completed):
            for m_i, (m, c) in enumerate(zip(self._machines, connections)):
                if completed[m_i]:
                    continue

                _, stdout, _ = c.exec_command("ls")
                stdout = str(stdout.read())

                associated_chunks = list(association[m["hostname"]])
                offset = 0
                for c_i, chunk in enumerate(associated_chunks):
                    output_name = self._get_output_name(chunk)
                    if output_name in stdout:
                        self._retrieve_pkl(c, output_name)
                        try:
                            outputs[self._get_chunk_id(chunk)] = self._read_pkl(output_name)
                            association[m["hostname"]].remove(chunk)
                            del launch_time[m["hostname"]][c_i - offset]
                            c.exec_command(f"rm {chunk}")
                            c.exec_command(f"rm {output_name}")
                            offset += 1
                        except Exception as e:
                            print(f"An exception occurred: {e}")
                            # Make it look like a timeout so that it will be restarted
                            launch_time[m["hostname"]][c_i - offset] = -float("inf")

                if len(association[m["hostname"]]) == 0:
                    completed[m_i] = True

                # Check for stalled jobs
                for t_i, ltime in enumerate(launch_time[m["hostname"]]):
                    if time() - ltime > int(m["maximum_time"]):
                        # Launch it again, probably the run stalled
                        self._start_job(c, m, fname, association[m["hostname"]][t_i], cname)
                        launch_time[m["hostname"]][t_i] = time()
            print(f"{np.sum([len(x) for x in association.values()])} jobs remaining", end=" "*20 + "\r")
            sleep(1)
        for m, c in zip(self._machines, connections):
            # Clean pickles in case of re-launches
            if "kill_cmd" in m:
                c.exec_command(m["kill_cmd"])
            c.exec_command("rm *_*.pkl")
            c.close()

        unpacked_outputs = []
        for o in outputs:
            unpacked_outputs.extend(o)
        return unpacked_outputs


class EnvBasedMapper(Mapper):

    """
    This is the base class for mappers that depend on the environment
    for the mapping process
    """

    def __init__(self, n_samples):
        """
        Initializes the mapper

        :n_samples: the samples to store in the archive
        """
        Mapper.__init__(self)

        self._n_samples = n_samples
        self._archive = None

    def _initialize_archive(self, config):
        """
        Initializes the archive with a set of random samples sampled from an
        episode.
        :returns: A list
        """
        # TODO: Research question: is an episode enough? <09-12-21, Leonardo Lucio Custode> #
        fit_cfg = config["Fitness"]
        env = TaskMetaClass.get(fit_cfg["name"])(**fit_cfg["kwargs"])
        observations = [env.reset()]

        done = False

        while not done:
            obs, _, done = env.step(env.sample_action())
            observations.append(obs)

        samples = np.random.choice(
            [*range(len(observations))],
            size=self._n_samples,
            replace=False
        )

        return np.array([observations[i] for i in samples])

    def _evaluate(self, pipelines, config):
        return self._mapper.map(
            utils.get_output_pipeline,
            inputs=[(p, self._archive) for p in pipelines],
            config
        )


class KNNAcceleratedMapper(EnvBasedMapper, metaclass=MapperMetaClass):

    """
    This is a wrapper for mappers which approximate, when possible, the
    results with the KNN algorithm
    """

    def __init__(
         self,
         mapper_name,
         k,
         max_dist,
         n_samples,
         **mapper_args
                    ):
        """
        Initializes the mapper

        :mapper_name: The name of the mapper to wrap
        :k: The number of neighbors to use
        :max_dist: The maximum distance from the neighbors
        :n_samples: The number of samples to use for the computation of
                    the outputs
        :**mapper_args: The args for the wrapped mapper
        """
        Mapper.__init__(self, n_samples)

        self._mapper = MapperMetaClass.get(mapper_name)(**mapper_args)
        self._k = k
        self._max_dist = max_dist
        self._sol_space = []
        self._archived_fitnesses = []

    def _knn_regression(self, outputs):
        """
        Performs the regression on the outputs of the pipeline
        """
        approximation = None

        # Checself._k if there are enough points
        if len(self._sol_space) < self._k:
            return approximation

        # Compute the distances
        distances = np.sum(np.abs(self._sol_space - outputs), axis=1)

        # Sort by nearest
        nearest = np.argsort(distances)

        # Simplest case: distance with nearest = 0
        if distances[nearest[0]] == 0:
            return self._archived_fitnesses[nearest[0]]

        # Check if we can approx it by using KNN
        if distances[nearest[self._k-1]] <= self._max_dist:
            sum_weights = 0
            approximation = 0

            for i in nearest[:self._k]:
                alpha = 1/distances[i]
                sum_weights += alpha
                approximation += alpha * self._archived_fitnesses[i]
            approximation /= sum_weights
        return approximation

    def map(self, function, inputs, config):
        """
        Performs mapping

        :function: The function to map
        :inputs: The list of inputs to pass to the function
        :config: The dictionary containing the parameters
        :returns: A list of outputs returned by the function for each input
        """
        t = time()
        if self._archive is None:
            # Then, initialize the archive of samples
            self._archive = self._initialize_archive(config)

        filtered = [False] * len(inputs)

        new_inputs = []
        temp_fits = [None] * len(inputs)
        temp_pipes = [None] * len(inputs)
        temp_outputs = self._evaluate(inputs, config)

        for i, p in enumerate(inputs):
            temp_pipes[i] = p

            fitness = self._knn_regression(temp_outputs[i])

            if fitness is not None:
                filtered[i] = True
                temp_fits[i] = (fitness, p)
            else:
                new_inputs.append(p)

        print(f"KNN took {time() - t} seconds.")
        print(f"Mapper filtered {sum(filtered)} samples.")
        computed_fitnesses = self._mapper.map(function, new_inputs, config)

        fit_idx = 0

        for i, fits in enumerate(temp_fits):
            if fits is None:
                temp_fits[i] = computed_fitnesses[fit_idx]

                if len(self._sol_space) == 0:
                    self._sol_space = temp_outputs[i].reshape(1, -1)
                else:
                    self._sol_space = np.r_[self._sol_space, temp_outputs[i]]

                self._archived_fitnesses.append(computed_fitnesses[fit_idx][0])
                # TODO: Implement a class EvaluationResult that is returned by the fitness function <09-12-21, Leonardo Lucio Custode> #

                fit_idx += 1
        return temp_fits


class DBSCANAcceleratedMapper(EnvBasedMapper, metaclass=MapperMetaClass):

    """
    This class accelerates the mapping by performing DBSCAN clustering
    and evaluating only the centroid of the cluster"""

    def __init__(
        self,
        mapper_name,
        eps,
        decay,
        n_samples,
        min_samples,
        **mapper_args
    ):
        """
        Initializes the mapper

        :eps: The initial radius for clustering
        :decay: The decay for the radius i.e., at step k, eps_k = eps * decay^k
        :n_samples: The number of samples to store in the archive of inputs
        :min_samples: The minimum number of points that defines a cluster

        """
        EnvBasedMapper.__init__(self, n_samples)

        self._mapper = MapperMetaClass.get(mapper_name)(**mapper_args)
        self._eps = eps
        self._decay = decay
        self._n_samples = n_samples
        self._min_samples = min_samples
        self._n_calls = 0

    def map(self, function, inputs, config):
        """
        Performs mapping

        :function: The function to map
        :inputs: The list of inputs to pass to the function
        :config: The dictionary containing the parameters
        :returns: A list of outputs returned by the function for each input
        """
        t = time()
        if self._archive is None:
            # Then, initialize the archive of samples
            self._archive = self._initialize_archive(config)

        fitnesses = np.ones(len(inputs)) * float("-inf")
        dependency = [None] * len(inputs)

        temp_outputs = self._evaluate(inputs, config)
        temp_outputs = np.array(temp_outputs)

        dbs = DBSCAN(
            eps=self._eps * (self._decay ** self._n_calls),
            min_samples=self._min_samples,
            n_jobs=-1
        )

        clusters = dbs.fit_predict(temp_outputs)

        cluster_ids = np.unique(clusters)

        for c in cluster_ids:
            if c != -1:
                min_dist = float("inf")
                argmin = None

                # Find points in the same cluster
                indices = np.where(clusters == c)[0]

                # Find centroid
                for i in indices:
                    cur_dist = np.mean(np.linalg.norm(
                        temp_outputs[i] - temp_outputs[indices],
                        axis=1
                    ))
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        argmin = i

                # Assign dependencies
                for i in indices:
                    if i != argmin:
                        dependency[i] = c

        to_evaluate = [i for i, v in enumerate(dependency) if v is None]
        pip_to_eval = list(map(lambda x: inputs[x], to_evaluate))

        print(f"DBSCAN took {time() - t} seconds.")
        print(f"Mapper filtered {sum(x is not None for x in dependency)} pts.")
        computed_fitnesses = self._mapper.map(function, pip_to_eval, config)

        fitnesses[to_evaluate] = computed_fitnesses

        for index, d in enumerate(dependency):
            if d is not None:
                fitnesses[index] = fitnesses[d]

        return fitnesses
