"""
This packages manages experiment settings.
"""
from .experiment import Experiment
from .sample_experiment import SampleExperiment
from .explicit_questions_experiment import ExplicitQuestionsExperiment

def get_experiment(experiment_name: str, *args, **kwargs) -> Experiment:
    if experiment_name == "sample_experiment":
        return SampleExperiment(*args, **kwargs)
    elif experiment_name == "explicit_questions_experiment":
        return ExplicitQuestionsExperiment(*args, **kwargs)
    else:
        raise ValueError(f"Unknown experiment {experiment_name}")
