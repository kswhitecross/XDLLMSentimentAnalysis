"""
This file includes code to load huggingface models
"""
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_hf_model(model_name: str, use_flash_attn: bool=True):
    hf_token = os.environ.get("HF_ACCESS_TOKEN", default=None)
    if hf_token is None:
        raise ValueError("HF_ACCESS_TOKEN environment variable not set.  Please create a HuggingFace access token and"
                         "set HF_ACCESS_TOKEN to it")
    if use_flash_attn:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto',
                                                     attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM(model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
