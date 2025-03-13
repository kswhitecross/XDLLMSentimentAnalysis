"""
This file contains the Experiment Base class, which defines the methods an experiment must have.
"""
from abc import ABC, abstractmethod
from typing import Generator, Any


class Experiment(ABC):
    """
    Experiment base class.
    """
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name

    @abstractmethod
    def _get_experiment_generator(self) -> Generator[dict[str, Any], None, None]:
        """
        This method creates the experiment generator, which yields dictionaries containing all the relevant information
         to be logged and passed to the model.
        """
        pass

    def create_experiment(self) -> Generator[dict[str, Any], None, None]:
        # create the experiment
        exp_gen = self._get_experiment_generator()

        # create a new generator that wraps it, and verifies it has the necessary keys
        def wrapped_experiment():
            for exp in exp_gen:
                # make sure the experiment dict generated has the keys we need
                assert 'context' in exp
                assert 'max_gen_tokens' in exp
                yield exp
        return wrapped_experiment()

    @abstractmethod
    def evaluate_results(self, result_dict: dict[str, Any]):
        """
        This method gets the result dict containing the generated output `model_answer`, evaluates it, and adds the
         evaluation results to be logged to `result_dict`
        """
        pass

    @property
    @abstractmethod
    def n_experiments(self):
        """
        The length of the experiment generator, aka the number of experiments.
        :return:
        """
        pass

    @staticmethod
    def tqdm_metrics_dict(result_dict: dict[str, Any]) -> dict[str, Any]:
        """
        This method returns a dictionary of any metrics to be logged at the tqdm progress bar.  Optionally override it
         for each experiment.
        """
        return {}
