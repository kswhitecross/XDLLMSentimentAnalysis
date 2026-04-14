import matplotlib.pyplot as plt
import numpy as np
import json

# Example: 12 models, each with a distribution over 5 scores
# Replace with your actual data
model_names = ["llama_1b", "llama_3b", "llama_8b", "llama_70b"]
model_contexts = ["original", "med", "long"]
score_labels = [1, 2, 3, 4, 5]

# Each row corresponds to a model's distribution over 5 scores
distributions = np.zeros((12,5))  # dummy normalized data
for x, model_context in enumerate(model_contexts):
        for y, model_name in enumerate(model_names):
            file1 = open(f'distributions/{model_context}_{model_name}_sentiment_dist.json', 'r')

            json1 = json.load(file1)

            dist = np.array([json1["1"], json1["2"], json1["3"], json1["4"], json1["5"]])
            dist = dist / np.sum(dist)
            distributions[x*4 + y] = dist

model_names = ["llama_1b", "llama_3b", "llama_8b", "llama_70b"]
model_contexts = ["short", "med", "long"]

merged_names = []
for context in model_contexts:
    for model in model_names:
        merged_names.append(f"{context}_{model}")
# Create bar positions
x = np.arange(len(merged_names))
width = 0.15  # Width of each bar

fig, ax = plt.subplots(figsize=(14, 6))

# Plot each score category
for i, score in enumerate(score_labels):
    ax.bar(x + i*width, distributions[:, i], width, label=f'Score {score}')

# Labeling
ax.set_ylabel('Proportion')
ax.set_title('Score Distributions Across 12 LLMs')
ax.set_xticks(x + width*2)
ax.set_xticklabels(merged_names, rotation=45, ha='right')
ax.legend(title="Score")
plt.tight_layout()
plt.savefig("sentiment_model_dist.png")