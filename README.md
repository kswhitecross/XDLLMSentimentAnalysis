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

### Debugging

To see the input/output of the at each step of the experiment, set `verbose` to `True`.  For example,

`python main.py verbose True`

Additionally, to stop the experiment after a single run, use the `short_circuit` option:

`python main.py verbose True short_circuit True`

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

## Major Requriments 
- [`torch`](https://pytorch.org/)
- [`transformers`](https://huggingface.co/docs/transformers/installation)
- [`accelerate`](https://pypi.org/project/accelerate/)
- [`yacs`](https://pypi.org/project/yacs/)
- [`tqdm`](https://tqdm.github.io/)
- [`bitsandbytes`](https://huggingface.co/docs/bitsandbytes/main/en/installation)

### Flash Attention

For major speedups on longer context experiments, flash attention reduces the memory impact and improves performance.  However, it is complex to install, so only recommended for systems that will be used to run major experiments.

- [`flash-attn`](https://github.com/Dao-AILab/flash-attention) (gpu-only)

## Data
### Download Links
- [Books](https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset?resource=download)
- [YouTube](https://www.kaggle.com/datasets/atifaliak/youtube-comments-dataset)
- [News](https://www.kaggle.com/datasets/aryansingh0909/nyt-articles-21m-2000-present)

Don't forget to change FILE_PATH in datasets/books.py, datasets/youtube.py and datasets/news.py with the path to your data.

## Sentiment Analysis

A LLM-based sentiment analysis of each model response can be performed by using `sentiment_analysis.py` and specifying the folder containing each run you want to perform sentiment analysis on.  For example:

`python sentiment_analysis.py --runs_dir [path/to/your/runs]`

This will produce a `sentiment.jsonl` file alongside `results.jsonl`, with each line containing a score and justification.

## Pyschobench

[TODO]



