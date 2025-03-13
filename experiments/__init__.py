"""
This packages manages experiment settings.
"""
from .experiment import Experiment
from .sample_experiment import SampleExperiment


def get_experiment(experiment_name: str, *args, **kwargs) -> Experiment:
    if experiment_name == "sample_experiment":
        return SampleExperiment(*args, **kwargs)
    else:
        raise ValueError(f"Unknown experiment {experiment_name}")
