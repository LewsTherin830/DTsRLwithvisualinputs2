import numpy as np


class TaskMetaClass(type):
    _registry = {}

    def __init__(cls, clsname, bases, methods):
        super().__init__(clsname, bases, methods)
        TaskMetaClass._registry[cls.__name__] = cls

    @staticmethod
    def get(class_name):
        """
        Retrieves the class associated to the string

        :class_name: The name of the class
        :returns: A class
        """
        return TaskMetaClass._registry[class_name]


def fitness_function(pipeline, config):
    """
    Evaluates the fitness of the pipeline

    :pipeline: The pipeline to evaluate
    :config: The dictionary containing the parameters.
        Must contain a Fitness item which contains:
            - name: The name of the task
            - kwargs: The kwargs for the task
            - episodes: The number of episodes
    :returns: A tuple (fitness, trained pipeline)
    """
    fit_dict = config["Fitness"]

    name = fit_dict["name"]
    kwargs = fit_dict["kwargs"]
    episodes = fit_dict["episodes"]
    task = TaskMetaClass.get(name)(**kwargs)

    cumulative_rewards = []

    for i in range(episodes):
        pipeline.new_episode()
        cumulative_rewards.append(0)
        done = False
        task.seed(i)
        obs = task.reset()

        while not done:
            output = pipeline.get_output(obs)
            obs, rew, done = task.step(output)
            pipeline.set_reward(rew)

            cumulative_rewards[-1] += rew
    task.close()
    return np.mean(cumulative_rewards), pipeline
