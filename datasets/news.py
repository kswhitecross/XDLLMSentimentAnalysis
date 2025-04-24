"""
This file contains implementation of the news articles dataset.
"""
from torch.utils.data import Dataset
import os
import pandas as pd
import time

# specify dataset file path (change to your own)
# link to download dataset: https://www.kaggle.com/datasets/aryansingh0909/nyt-articles-21m-2000-present
FILE_PATH = "/Users/polinapetrova/data/nyt-metadata.csv"


class NewsDataset(Dataset):
    """
    Class to read data relating to NYT news articles.   #Polina
    """
    def __init__(self, file_path=FILE_PATH, chunksize=100000):
        """
        This dataset is very large. 
        It contains 2.1M instances, so it needs to be processed in chunks.
        """
        assert os.path.exists(file_path), f"File not found: {file_path}"
        
        # attributes to use (can be modified)
        # columns i didn't include:
        # web url
        # news desk (dept responsible for article)
        # byline (article authors)
        # word count
        # id
        # uri
        # print page
        # print section
        # source (we know all the articles came from the NYT)
        # document type (documents type are all articles)
        # section name
        # multimedia
        self.keep_columns = [
            'abstract', 'snippet', 'lead_paragraph', 
            'headline', 'keywords', 'pub_date', 
            'type_of_material', 'subsection_name'
        ]
        # to consider:
        # 'subsection_name' column contains many NaaN rows but indicates what topic the article relates to (ex. politics)
        # 'type_of_material' indicates type of article (ex. news)
        # dataset contains articles from 2000 to 2025, so we can filter which dates to keep (maybe by when the reddit posts were created)
        # i was unsure whether to include all 3: abstract, snippet and lead_paragraph, we can discuss later
        
        # empty list to store chunks
        chunks = []
        
        # read in chunks and append to the list
        for chunk in pd.read_csv(file_path, usecols=self.keep_columns, chunksize=chunksize):
            # since our focus is on news articles only, keep only those instances
            filtered = chunk[chunk['type_of_material'] == "News"]
            chunks.append(filtered)
        
        # concatenate chunks into a single dataframe
        self.data = pd.concat(chunks, ignore_index=True)

        self.data['pub_date'] = pd.to_datetime(self.data['pub_date']).dt.strftime('%Y-%m-%d')


    def __getitem__(self, idx):
        """
        Retrieves item at specified index from the dataset.
        :param idx: index of item
        :return: dictionary of item's attributes
        """
        row = self.data.iloc[idx]
        
        # excluded type of material because we know they're all news
        return {
            "headline": row['headline'],
            "abstract": row['abstract'], 
            "snippet": row['snippet'], 
            "lead_paragraph": row['lead_paragraph'], 
            "keywords": row['keywords'], 
            "pub_date": row['pub_date'],  
            "topic": row['subsection_name']
        }
    
    def __len__(self):
        return len(self.data)


# # example of printing out an article's info
# start_time = time.time()
# model = NewsDataset()
# print("Number of articles:", model.__len__())
# print("Most recent article:", model.__getitem__(-1))
# end_time = time.time()
# print(f"Execution time: {end_time - start_time:.2f} seconds")