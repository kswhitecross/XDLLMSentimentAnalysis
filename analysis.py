import os
import json
import numpy as np
from scipy.special import kl_div
from transformers import pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# HF's prebuild sentiment analysis pipeline
sentiment_model = pipeline("sentiment-analysis",  return_all_scores=True)

def get_sentiment_distro(text):
    result = sentiment_model(text)
    return result

def get_average_distro_from_samples(subfolder_path):
    # This will accumulate the distros across samples for a specific experiment 
    sentiment_sums = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    total_rows = 0
    
    # Gets the results for that experiment to aggregate
    results_file = os.path.join(subfolder_path, "results.jsonl")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            for line in f:
                data = json.loads(line)
                model_answer = data.get("model_answer", "")
                if model_answer:

                    sentiment_distro = get_sentiment_distro(model_answer)
                    # print(sentiment_distro)

                    # Take the only element in the array, produces an array because you could batch multiple at once
                    for prob in sentiment_distro[0]:
                        if prob["label"] == "POSITIVE":
                            sentiment_sums['POSITIVE'] += prob["score"]
                        elif prob["label"] == "NEGATIVE":
                            sentiment_sums['NEGATIVE'] += prob["score"]
                        else:
                            sentiment_sums['NEUTRAL'] += prob["score"]

                    total_rows += 1

    # Given that there were actually samples to average over, get the average distro
    if total_rows > 0:
        avg_sentiment_dist = {label: sentiment_sums[label] / total_rows for label in sentiment_sums}
        
        # TODO: Normalize it so it is valid for KL divergence? Can do this a different way, idk if this is legit. 
        total_sum = sum(avg_sentiment_dist.values())
        normalized_dist = {label: avg_sentiment_dist[label] / total_sum for label in avg_sentiment_dist}
        return normalized_dist
    return None
    

def plot_heatmaps_of_shifts(subfolder_sentiment_distributions, control_sentiment_distribution):
    rows = []
    for pair, dist in subfolder_sentiment_distributions.items():
        parts = pair.split('_in_context_')
        in_context = parts[1].split('_inquiry_')[0]
        inquiry = parts[1].split('_inquiry_')[1].split('_')[0]

        shift = calculate_shift(dist, control_sentiment_distribution)

        rows.append({
            'In-Context Domain X': in_context,
            'Inquiry Domain Y': inquiry,
            'POSITIVE': shift['POSITIVE'],
            'NEUTRAL': shift['NEUTRAL'],
            'NEGATIVE': shift['NEGATIVE']
        })

    df = pd.DataFrame(rows)

    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    for idx, sentiment in enumerate(['POSITIVE', 'NEUTRAL', 'NEGATIVE']):
        pivot = df.pivot_table(index='In-Context Domain X', columns='Inquiry Domain Y', values=sentiment)
        sns.heatmap(pivot, annot=True, cmap='coolwarm', center=0, linewidths=0.5, ax=axs[idx], cbar_kws={'label': 'Shift'})
        axs[idx].set_title(f'{sentiment} Sentiment Shift')
        axs[idx].set_xlabel('Inquiry Domain Y')
        if idx == 0:
            axs[idx].set_ylabel('In-Context Domain X')

    plt.suptitle('Sentiment Shifts from Control Distribution', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_heatmaps_of_kl_divergence(subfolder_sentiment_distributions, control_sentiment_distribution):
    kl_rows = []
    for pair, dist in subfolder_sentiment_distributions.items():
        parts = pair.split('_in_context_')
        in_context = parts[1].split('_inquiry_')[0]
        inquiry = parts[1].split('_inquiry_')[1].split('_')[0]

        kl = np.sum(calculate_kl_divergence(dist, control_sentiment_distribution))
        kl_rows.append({
            'In-Context Domain X': in_context,
            'Inquiry Domain Y': inquiry,
            'KL Divergence': kl
        })

    df = pd.DataFrame(kl_rows)
    pivot = df.pivot_table(index='In-Context Domain X', columns='Inquiry Domain Y', values='KL Divergence')

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, cmap='YlGnBu', linewidths=0.5, cbar_kws={'label': 'KL Divergence'})
    plt.title('KL Divergence from Control Distribution')
    plt.xlabel('Inquiry Domain Y')
    plt.ylabel('In-Context Domain X')
    plt.tight_layout()
    plt.show()

def calculate_kl_divergence(p, q):
    labels = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
    p_vector = np.array([p.get(label, 0) for label in labels])
    q_vector = np.array([q.get(label, 0) for label in labels])
    
    # Returns array of ELEMENT-WISE kl divergence for a more fine grained peek at what contributes 
    # to the drift in distro
    return kl_div(p_vector, q_vector)


# This is just like, trying to get a directional change with + and - values ? 
def calculate_shift(in_context_dist, control_dist):
    shift = {label: in_context_dist.get(label, 0) - control_dist.get(label, 0) for label in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']}
    return shift
    

def main(cross_domain_run_folder_path, control_subfolder_name):
    subfolder_sentiment_distributions = {}
    control_sentiment_distribution = None

    # Collect the sentiment distros for the full cross-domain run 
    for subfolder in os.listdir(cross_domain_run_folder_path):
        subfolder_path = os.path.join(cross_domain_run_folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            sentiment_dist = get_average_distro_from_samples(subfolder_path)
            if sentiment_dist is not None:
                subfolder_sentiment_distributions[subfolder] = sentiment_dist

    control_path = os.path.join('runs', 'implicit', 'control', control_subfolder_name)
    if os.path.isdir(control_path):
        control_dist = get_average_distro_from_samples(control_path)
        # print(control_dist)
        if control_dist is not None:
            control_sentiment_distribution = control_dist

    # Calculate KL divergence between subfolder sentiment distros from this full cross-domain run
    for in_context_subfolder, in_context_dist in subfolder_sentiment_distributions.items():
        # Only calculating the divergence between the inquiry domain and the rest of the domains
        if in_context_subfolder != control_subfolder_name:
            element_wise_kl_div = calculate_kl_divergence(in_context_dist, control_sentiment_distribution)
            print(f"==== COMPARING {in_context_subfolder} and {control_subfolder_name} ====")
            print(f"Element-wise divergences between {in_context_subfolder} and {control_subfolder_name}: {element_wise_kl_div}")
            print(f"KL Divergence between {in_context_subfolder} and {control_subfolder_name}: {np.sum(element_wise_kl_div)}")
            print(f"Directional shift from control is { calculate_shift(in_context_dist, control_sentiment_distribution)}")
            print("==========================================================\n")

    # TY chatgpt for helping set up these labeled heatmaps 
    plot_heatmaps_of_shifts(subfolder_sentiment_distributions=subfolder_sentiment_distributions,
                            control_sentiment_distribution=control_sentiment_distribution)
    plot_heatmaps_of_kl_divergence(subfolder_sentiment_distributions=subfolder_sentiment_distributions,
                            control_sentiment_distribution=control_sentiment_distribution)
if __name__ == "__main__":

    # TODO obviously change this based on the run we are using to compare/generate plots
    cross_domain_run_folder_path = 'runs/implicit/experimental/20b8e0af06cb4c3abff3c71a419bedee'
    control_folder_name = 'implicit_control_sample_split2_af51e91f847a499f9f03afc1400ba41f'

    main(cross_domain_run_folder_path=cross_domain_run_folder_path, control_subfolder_name=control_folder_name)

