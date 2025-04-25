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
    # Save all of the contexts?
    cfg.save_context = True
    # Print out model inputs / outputs?
    cfg.verbose = False
    # Stop after a single run
    cfg.short_circuit = False
    # These will be set later
    cfg.run_id = None
    cfg.name_id = None
    cfg.save_path = None
    cfg.dont_save = None

    # ====== Experiment Settings ======
    # these settings will be passed as keyword arguments directly to get_experiment
    cfg.exp = CfgNode()
    # which experiment to run?
    # 'sample_experiment' to run a sample, experiment for debugging purposes
    cfg.exp.name = "sample_experiment"
    # names and splits for the two datasets used
    # 'sample' for a very basic, sample dataset
    cfg.exp.d1_name = "sample"
    cfg.exp.d2_name = "sample"
    # # splits for each of the two datasets used
    # # for 'sample', options are 'split1', 'split2';q
    cfg.exp.d1_split = "split1"
    cfg.exp.d2_split = "split2"
    # legacy max generate
    # NO LONGER USED
    cfg.exp.max_generate = 256

    # specifies how many samples from the in-context domain to provide to the freshly initialized model
    cfg.exp.num_in_context_samples = 2
    # specifies how many inquiries to address before cutting it off for the second domain...
    # if None then it responds to the full set unless short-circuited
    cfg.exp.num_inquiry_samples = None
    # For the same prompt, specifies how many times to regenerate the randomly sampled output
    cfg.exp.num_outputs_per_prompt = 1
    # Specifies the exact prompt to use 
    cfg.exp.prompt_name = "SampleExperimentPrompt.txt"
    # number of comments to include per reddit post
    cfg.exp.num_comments = 10

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
    # Quantize the model, only matters if model.use_flash_attn is False
    cfg.model.quantize = False
    # Model sampling parameters
    # Passed as a dictionary to update model.generation_config
    cfg.model.gen = CfgNode()
    # Whether or not to do any sampling
    # If false, will do greedy decoding
    cfg.model.gen.do_sample = True
    # temperature for sampling
    cfg.model.gen.temperature = 0.6
    # probability threshhold for nucleus sampling
    cfg.model.gen.top_p = 0.90
    # maximum of new tokens to generate
    cfg.model.gen.max_new_tokens = 2000

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

# create the default config if it doesn't exist
default_config_path = os.path.join('configs', 'default.yaml')
if not os.path.exists(default_config_path):
    print("Default config file does not exist.  Creating...")
    with open(default_config_path, 'w') as f:
        f.write(CFG.dump())
