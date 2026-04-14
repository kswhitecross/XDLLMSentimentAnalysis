from scipy.stats import chisquare
from scipy.special import kl_div
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    model_names = ["llama_1b", "llama_3b", "llama_8b", "llama_70b"]
    model_contexts = ["original", "long"]

    merged_names = []
    for context in model_contexts:
        for model in model_names:
            merged_names.append(f"{context}_{model}")

    #8x8 because 8 total configurations
    kl_divs = np.zeros((8,8))

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

                    dist_1_prob = dist_1 / dist_1.sum()
                    dist_2_prob = dist_2 / dist_2.sum()

                    kl_divergence = np.sum(kl_div(dist_1_prob, dist_2_prob))

                    if a == 0 and c == 0:
                        if d == 1 and (b == 2 or b == 3):
                            print(f'distributions/{model_context_2}_{model_name_2}_sentiment_dist.json')

                    kl_divs[a + 4*c][b + 4*d] = kl_divergence

    log_kl_divs = np.around(np.log(kl_divs), 2)
    fig, ax = plt.subplots()
    im = ax.imshow(log_kl_divs)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(range(len(merged_names)), labels=merged_names,
                rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(merged_names)), labels=merged_names)

    # Loop over data dimensions and create text annotations.
    for i in range(len(merged_names)):
        for j in range(len(merged_names)):
            text = ax.text(j, i, log_kl_divs[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Natural log KL divergence between sentiment score distributions")
    fig.tight_layout()
    plt.savefig("kl_div.png")

    print(kl_divs[0][1])
    print(kl_divs[1][0])

    file1 = open(f'distributions/original_llama_1b_sentiment_dist.json', 'r')
    file2 = open(f'distributions/original_llama_3b_sentiment_dist.json', 'r')

    json1 = json.load(file1)
    json2 = json.load(file2)

    dist_1 = np.array([json1["1"], json1["2"], json1["3"], json1["4"], json1["5"]])
    dist_2 = np.array([json2["1"], json2["2"], json2["3"], json2["4"], json2["5"]])

    dist_1_prob = dist_1 / dist_1.sum()
    dist_2_prob = dist_2 / dist_2.sum()

    kl_divergence = np.sum(kl_div(dist_2_prob, dist_1_prob))

    print(kl_divergence)

def js_divergence(p, q):
    # Normalize the distributions to sum to 1 (turn into probability distributions)
    p = np.array(p) / sum(p)
    q = np.array(q) / sum(q)
    
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