# XDLLMSentimentAnalysis
Cross-Domain Sentiment Analysis with LLMs

## Usage

To run an experiment, specify a config file (default: `configs/default.yaml`).

`python main.py --config configs/{config} [arg1 val1]...`

Any arguments in the config can be manually overridden by specifying their name and the value to set, after specifying the config. For example to specify a different model than the one in the config:

`python main.py --config configs/default.yaml model.name 'meta-llama/Llama-3.2-1B'`

### Running on CPU

To run experiments on the CPU, disable flash attention by specifying `model.use_flash_attn False`.  For example,

`python main.py model.use_flash_attn False`

## Project hierarchy

`data/` where any custom data that needs to be downloaded is stored, excluding cached data from third-party datasets.

`datasets/` our implementations/wrappers of datasets we're using.

`experiments/` our implementations of each different experiment we want to run.  

`runs/` where outputs/experimental results are stored.  Each experiment is stored in a folder named with its id, with output results as JSONL (JSON-list) files.

`prompts/` contains prompt templates to be read in and formatted.

`models/` code that loads the models we want to use.

`configs/` stores YAML config files.  Each config completely describes an experiment.

`notebooks/` stores `.ipynb` notebooks, used for data exploration and visualization.

`config.py` manages and defines the `CFG` object, which contains all experiment configurations.

`main.py` main function that reads a config file and creates the configuration, runs an experiment, and writes the results to a folder.

## Requriments (that I can think of)
- [`torch`](https://pytorch.org/)
- [`transformers`](https://huggingface.co/docs/transformers/installation)
- [`accelerate`](https://pypi.org/project/accelerate/)
- [`yacs`](https://pypi.org/project/yacs/)
- [`tqdm`](https://tqdm.github.io/)
- [`bitsandbytes`](https://huggingface.co/docs/bitsandbytes/main/en/installation)

### Flash Attention

For major speedups on longer context experiments, flash attention reduces the memory impact and improves performance.  However, it is complex to install, so only recommended for systems that will be used to run major experiemnts.

- [`flash-attn`](https://github.com/Dao-AILab/flash-attention) (gpu-only)