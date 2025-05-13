import json
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


model_names = ["llama_1b.jsonl", "llama_3b.jsonl", "llama_8b.jsonl", "llama_70b.jsonl"]
model_contexts = ["original", "med", "long"]

for context in model_contexts:
    for model in model_names:
        results = []
        with open(f'/Users/ajchilds/Desktop/school/685/error_analysis/sentiment/{context}/{model}', 'r') as f: 
            for line in f:
                results.append(json.loads(line))

        sentiment_scores = []
        for result in results:
            if result["score"] is not None:
                sentiment_scores.append(int(result["score"]))
        frequency_dict = dict(Counter(sentiment_scores))
            
        print(len(sentiment_scores))
        plt.clf()
        # Create the histogram
        plt.hist(sentiment_scores, bins=np.arange(1, 7) - 0.5, color=(135/255, 206/255, 235/255), edgecolor='black')  # 'bins' specifies how many bins to divide the data into

        # Add titles and labels
        plt.title(f'{context} {model} sentiment score distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        # Show the plot
        plt.savefig(f"figs/{context}_{model}_sentiment_dist.png")
        with open(f"distributions/{context}_{model[:-6]}_sentiment_dist.json", "w") as file:
            json.dump(frequency_dict, file)
