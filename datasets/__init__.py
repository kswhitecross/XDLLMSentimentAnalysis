"""
Datasets package. Contains all custom implementations of datasets.
"""
# Import your datasets
from .books import BooksDataset
from .news import NewsDataset
from .youtube import YoutubeDataset
from .sample_dataset import SampleDataset  

def get_dataset(dataset_name: str, **kwargs):
    if dataset_name == "sample":
        return SampleDataset(**kwargs)
    elif dataset_name == "books":
        return BooksDataset(**kwargs)  
    elif dataset_name == "news":
        return NewsDataset(**kwargs)  
    elif dataset_name == "youtube":
        return YoutubeDataset(**kwargs)  
    else:
        raise ValueError(f"Unknown Dataset {dataset_name}")


# """
# Datasets package.  Contains all custom implementations of datasets.
# """
# from .reddit import RedditDataset
# from .sample_dataset import SampleDataset


# def get_dataset(dataset_name: str, **kwargs):
#     if dataset_name == "sample":
#         return SampleDataset(**kwargs)
#     else:
#         raise ValueError(f"Unknown Dataset {dataset_name}")
