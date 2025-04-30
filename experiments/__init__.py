"""
This packages manages experiment settings.
"""
from .experiment import Experiment
from .sample_experiment import SampleExperiment
from .explicit_questions_experiment import ExplicitQuestionsExperiment
from .implicit_questions_experiment import ImplicitQuestionsExperiment
from .reddit_implicit_questions_experiment import RedditImplicitQuestionsExperiment 
from .reddit_self_scoring_experiment import RedditSelfScoringExperiment

def get_experiment(experiment_name: str, *args, **kwargs) -> Experiment:
    if experiment_name == "sample_experiment":
        return SampleExperiment(*args, **kwargs)
    elif experiment_name == "explicit_questions_experiment":
        return ExplicitQuestionsExperiment(*args, **kwargs)
    elif experiment_name == "implicit_questions_experiment":
        return ImplicitQuestionsExperiment(*args, **kwargs)
    elif experiment_name == "reddit_implicit_questions_experiment":
        return RedditImplicitQuestionsExperiment(*args, **kwargs)
    elif experiment_name == "reddit_self_scoring_experiment":
        return RedditSelfScoringExperiment(*args, **kwargs)

    else:
        raise ValueError(f"Unknown experiment {experiment_name}")
