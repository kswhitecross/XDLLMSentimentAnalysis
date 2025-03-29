"""
This file contains implementation of the YouTube comments dataset.
"""
from torch.utils.data import Dataset
import os
import pandas as pd

# specify dataset file path (change to your own)
# link to download dataset: https://www.kaggle.com/datasets/atifaliak/youtube-comments-dataset
FILE_PATH = "/Users/polinapetrova/data/YoutubeCommentsDataSet.csv"


class YoutubeDataset(Dataset):
    """
    Class to read data relating to YouTube comments.    #Polina
    """
    def __init__(self, file_path=FILE_PATH, **kwargs):
        assert os.path.exists(file_path), f"File not found: {file_path}"
        
        # read youtube comment data (csv)
        self.data = pd.read_csv(file_path)

    def __getitem__(self, idx):
        """
        Retrieves item at specified index from the dataset.
        :param idx: index of item
        :return: dictionary of item's attributes
        """
        # kept only column of comment text
        # dataset also had a 'sentiment' column which i didn't include since
        # we'll be doing our own sentiment analysis
        comment = self.data.iloc[idx]['Comment']

        return {"comment": comment.strip()}
    
    def __len__(self):
        return len(self.data)
    