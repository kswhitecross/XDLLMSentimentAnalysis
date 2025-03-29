"""
This file contains implementation of the book summaries dataset.
"""
from torch.utils.data import Dataset
import os

# specify dataset file path (change to your own)
# link to download dataset: https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset?resource=download
FILE_PATH = "/Users/polinapetrova/data/booksummaries.txt"


class BooksDataset(Dataset):
    """
    Class to read data relating to book summaries.  #Polina
    """
    def __init__(self, file_path=FILE_PATH, **kwargs):
        assert os.path.exists(file_path), f"File not found: {file_path}"
        
        # read book summary data (txt)
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __getitem__(self, idx):
        """
        Retrieves item at specified index from the dataset.
        :param idx: index of item
        :return: dictionary of item's attributes
        """
        line = self.data[idx].strip().split('\t')

        # attributes to use (can be modified)
        # columns i didn't include:
        # book id
        # publishing date
        title = line[2]
        author = line[3]
        genres = line[5]
        summary = line[6]

        return {
            "title": title,
            "author": author,
            "genres": genres,
            "summary": summary
        }
    
    def __len__(self):
        return len(self.data)
