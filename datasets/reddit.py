"""
This file contanis the implementation of the reddit dataset.
"""
from torch.utils.data import Dataset


class RedditDataset(Dataset):
    """
    Description here.  #TODO Andrew
    """
    def __init__(self, subreddit: str):
        self.subreddit = subreddit

    def __getitem__(self, idx):
        """
        Returns the item at index idx.  Returns it as a dict containing ...
        :param idx:
        :return:
        """
        return {}

    def __len__(self) -> int:
        return 0
