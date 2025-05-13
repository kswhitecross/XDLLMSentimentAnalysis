from scipy.stats import chisquare
from scipy.special import kl_div
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    model_names = ["llama_1b", "llama_3b", "llama_8b", "llama_70b"]
    model_contexts = ["original", "med", "long"]
    #8x8 because 8 total configurations
    js_divs = np.zeros((12,12))

    #get every possible combination of two models: so 64 total:
    for c, model_context_1 in enumerate(model_contexts):
        for a, model_name_1 in enumerate(model_names):
            for d, model_context_2 in enumerate(model_contexts):
                for b, model_name_2 in enumerate(model_names):
                    file1 = open(f'distributions/{model_context_1}_{model_name_1}_sentiment_dist.json', 'r')
                    file2 = open(f'distributions/{model_context_2}_{model_name_2}_sentiment_dist.json', 'r')

                    json1 = json.load(file1)
                    json2 = json.load(file2)

                    dist_1 = np.array([json1["1"], json1["2"], json1["3"], json1["4"], json1["5"]])
                    dist_2 = np.array([json2["1"], json2["2"], json2["3"], json2["4"], json2["5"]])

                    js_div = js_divergence(dist_1, dist_2)

                    js_divs[a + 4*c][b + 4*d] = js_div
    model_names = ["llama_1b", "llama_3b", "llama_8b", "llama_70b"]
    model_contexts = ["short", "med", "long"]

    merged_names = []
    for context in model_contexts:
        for model in model_names:
            merged_names.append(f"{context}_{model}")

    log_js_divs = np.around(np.log(js_divs), 2)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(log_js_divs)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(range(len(merged_names)), labels=merged_names,
                rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(merged_names)), labels=merged_names)

    # Loop over data dimensions and create text annotations.
    for i in range(len(merged_names)):
        for j in range(len(merged_names)):
            text = ax.text(j, i, log_js_divs[i, j],
                        ha="center", va="center", color="w", fontsize=14)

    ax.set_title("Natural log JS divergence between sentiment score distributions")
    fig.tight_layout()
    plt.savefig("js_div.png")

    # print(js_divs[0][1])
    # print(js_divs[1][0])

def js_divergence(p, q):
    # Normalize the distributions to sum to 1 (turn into probability distributions)
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Compute the mean distribution
    m = 0.5 * (p + q)
    
    # Compute the KL divergences
    kl_p_m = kl_div(p, m).sum()
    kl_q_m = kl_div(q, m).sum()
    
    # Compute the Jensen-Shannon Divergence
    jsd = 0.5 * (kl_p_m + kl_q_m)
    
    return jsd

if __name__ == "__main__":
    main()