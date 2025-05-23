"""
This file contains the implementation of the reddit dataset.
"""
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os 
import ast

class RedditDataset(Dataset):
    """
    Top 10 subreddits, each with top 10 posts of the year (2025) and respective top 100 comments per post
    Specify the subreddit name as the split and how many comments to extract
    """
    def __init__(self, split: str, num_comments: int):
        if split is None:
            raise ValueError("subreddit name in RedditDataset constructor cannot be None")
        file_path = os.path.join('data', 'reddit_top_100_posts_scraped.csv')
        df = pd.read_csv(file_path)

        #filtering by subreddit name, so this should return only 10 rows
        self.data = df[df['subreddit'] == split]
        self.num_comments = num_comments
        # self.data = df.where(df['subreddit'] == split).dropna()

    def __getitem__(self, idx):
        """
        Returns the item at index idx.  Returns it as a dict containing ...
        :param idx:
        :return:
        """
        return {"content":{"post_title": self.data.iloc[idx]["post_title"], "post_body": self.data.iloc[idx]["post_body"], "comments": ast.literal_eval(self.data.iloc[idx]["comments"])[:self.num_comments]}}

    def __len__(self) -> int:
        return len(self.data)
    

# def main():
#     funny_dataset = RedditDataset("funny")
#     samp_idx = np.random.choice(len(funny_dataset), size=2, replace=False).tolist()
#     in_context_docs = "\n\n".join([f"PREVIOUS content chunk consumed:\n{funny_dataset[position]['content']}" for i, position in enumerate(samp_idx)])
#     print(in_context_docs)
# if __name__ == "__main__":
#     main()