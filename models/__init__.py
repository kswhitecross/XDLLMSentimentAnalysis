"""
This package handles creating and loading the LLMs needed to run the experiments.
"""
from .hf_models import get_hf_model


def get_model(model_type: str = 'hf', **kwargs):
    if model_type == "hf":
        return get_hf_model(**kwargs)
    else:
        raise ValueError(f"Unknown model type{model_type}")