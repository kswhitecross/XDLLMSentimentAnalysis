"""
Datasets package.  Contains all custom implementations of datasets.
"""
from .reddit import RedditDataset
from .sample_dataset import SampleDataset


def get_dataset(dataset_name: str, **kwargs):
    if dataset_name == "sample":
        return SampleDataset(**kwargs)
    elif dataset_name == "reddit":
        return RedditDataset(**kwargs)
    else:
        raise ValueError(f"Unknown Dataset {dataset_name}")
