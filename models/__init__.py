"""
This package handles creating and loading the LLMs needed to run the experiments.
"""
from .hf_models import get_hf_model


def get_model(model_type: str, model_name: str, use_flash_attn: bool=True, quantize: bool=False, **kwargs):
    if model_type == "hf":
        return get_hf_model(model_name, use_flash_attn, quantize=quantize, **kwargs)
    else:
        raise ValueError(f"Unknown model type / name {model_type}/{model_name}")