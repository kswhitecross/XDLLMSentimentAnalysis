"""
This file contanis the implementation of the reddit dataset.
"""
from torch.utils.data import Dataset
import pandas as pd


class RedditDataset(Dataset):
    """
    Description here.  #TODO Andrew
    """
    def __init__(self, subreddit: str):
        if subreddit is None:
            raise ValueError("subreddit name in RedditDataset constructor cannot be None")
        file_path = "./reddit_post_comments.csv"
        df = pd.read_csv(file_path)

        #filtering by subreddit name, so this should return only 10 rows
        self.data = df.where(df['subreddit'] == subreddit).dropna()

    def __getitem__(self, idx):
        """
        Returns the item at index idx.  Returns it as a dict containing ...
        :param idx:
        :return:
        """
        return {"subreddit_name": self.data.iloc[idx]["subreddit"], "post_title": self.data.iloc[idx]["post_title"], "comments": self.data.iloc[idx]["comments"]}

    def __len__(self) -> int:
        return len(self.data)
