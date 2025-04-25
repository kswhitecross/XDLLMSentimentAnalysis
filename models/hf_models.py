"""
This file includes code to load huggingface models
"""
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def clean_generation_kwargs(kwargs: dict):
    # if kwargs['do_sample'] is False, or doesn't exist...
    if not kwargs.get('do_sample', False):
        # ... Then remove top_p, top_k, temperature from kwargs
        kwargs = {k: v for k, v in kwargs.items()
                  if k not in ("top_p", "top_k", "temperature")}
    return kwargs


def get_hf_model(
        name: str = 'meta-llama/Llama-3.2-1B-Instruct',
        use_flash_attn: bool = True,
        quantize: bool = False,
        gen: dict = None,
        **kwargs):

    # get the access token
    hf_token = os.environ.get("HF_ACCESS_TOKEN", default=None)
    if hf_token is None:
        raise ValueError("HF_ACCESS_TOKEN environment variable not set.  Please create a HuggingFace access token and"
                         "set HF_ACCESS_TOKEN to it")

    # set up keyword arguments for creating the model
    model_kwargs = {
        'token': hf_token,
        'device_map': 'auto',
    }

    if use_flash_attn:
        model_kwargs['attn_implementation'] = 'flash_attention_2'

    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_kwargs['quantization_config'] = bnb_config
    else:
        model_kwargs['torch_dtype'] = torch.bfloat16

    # create the model
    model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)

    # add the generation parameters
    if gen is None:
        gen = {}
    model.generation_config.update(**clean_generation_kwargs(gen))

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, token=hf_token)
    return model, tokenizer
