"""
This file includes everything config related.  Including all config options and descriptions.
"""
from yacs.config import CfgNode
import uuid
import os


def get_new_id():
    return uuid.uuid4().hex


def get_config_defaults() -> CfgNode:
    """
    Creates a config node with the default values.

    All of the configuration options for all experiments are defined and initialized here.
    """
    cfg = CfgNode()

    # ====== Save Settings ======
    # runs will be saved to a folder `{save_dir}/{name}_{run_id}`, where run_id is a random uuid4
    # Name of this run configuration
    cfg.name = "default"
    # The root folder where all results will be saved
    cfg.save_dir = "runs/"
    # Save the context from each LLM forward pass?
    cfg.save_context = False
    # These will be set later
    cfg.run_id = None
    cfg.name_id = None
    cfg.save_path = None
    cfg.dont_save = None

    # ====== Experiment Settings ======
    cfg.exp = CfgNode()
    # which experiment to run?
    # options: ...
    cfg.exp.name = "Unk"
    # more experiment related settings here...

    # ====== Model Settings ======
    cfg.model = CfgNode()
    # model type
    # 'hf' for huggingface model
    cfg.model.type = 'hf'
    # Model name
    # Some options:
    # 'meta-llama/Llama-3.2-1B-Instruct'
    # 'meta-llama/Llama-3.2-3B-Instruct'
    # 'meta-llama/Llama-3.1-8B-Instruct'
    cfg.model.name = 'meta-llama/Llama-3.2-1B-Instruct'
    # Use flash attention, a GPU-only optimization of self-attention
    cfg.model.use_flash_attn = True

    return cfg


def update_config(p_cfg: CfgNode, config_path: str, arg_opts: list, dont_save: bool):
    """
    Updates the config based on the config at config_path, and command line arguments in arg_opts.
    """

    # merge the config from the file
    p_cfg.merge_from_file(config_path)

    # update options from command line arguments
    p_cfg.merge_from_list(arg_opts)

    # set paths and ids
    p_cfg.run_id = get_new_id()
    p_cfg.name_id = f'{p_cfg.name}_{p_cfg.run_id}'
    p_cfg.save_path = os.path.join(p_cfg.save_dir, p_cfg.name_id)
    p_cfg.dont_save = dont_save


def finalize_config(p_cfg: CfgNode):
    p_cfg.freeze()


CFG = get_config_defaults()
