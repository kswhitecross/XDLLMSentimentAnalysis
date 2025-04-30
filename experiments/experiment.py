"""
This file contains the Experiment Base class, which defines the methods an experiment must have.
"""
from abc import ABC, abstractmethod
from typing import Generator, Any
import json
import re

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
                yield exp
        return wrapped_experiment() 
    
    def evaluate_results(self, result_dict: dict[str, Any]):
        """
        This method gets the result dict containing the generated output `model_answer`, evaluates it, and adds the
            evaluation results to be logged to `result_dict`
        """
        ans = result_dict['model_answer']
        
        try:
            # Find first bracket, last bracket, anything in between (dotall allows newlines for multi line JSON)
            json_match = re.search(r'\{.*\}', ans, re.DOTALL)
            # Store it as is, without extracting innards just in case post-processing needed
            if json_match:
                json_data = json.loads(json_match.group())
                result_dict['json_sentiment_scoring'] = json_data
            else:
                # ans_with_ending = ans + "}"
                # hail_mary_json_match = re.search(r'\{.*\}', ans_with_ending, re.DOTALL)
                # if hail_mary_json_match:
                #     json_data = json.loads(hail_mary_json_match.group())
                #     result_dict['json_sentiment_scoring'] = json_data
                result_dict['json_sentiment_scoring'] = None
                print("No valid JSON found in the response.")


        except Exception as e:
            print("Trouble extracting JSON from model's anwer")
            result_dict['json_sentiment_scoring'] = None

        return

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
