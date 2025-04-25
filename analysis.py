import os
import json
import numpy as np
from scipy.special import kl_div
from transformers import pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# HF's prebuilt sentiment analysis pipeline
# TODO: handle longer sequences than 512 if necessary
# sentiment_model = pipeline("sentiment-analysis", truncation=True, top_k=None)
sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", truncation=True, max_length= 512, top_k=None)

def get_sentiment_distro_batch(text):
    result = sentiment_model(text)
    print(result)
    return result


def aggregate_dict_distros(distros):
    # # This will accumulate the distros across samples for a specific experiment 
    # sentiment_sums = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}

    # for distro in distros:
    #     for prob in distro:
    #         if prob["label"] == "POSITIVE":
    #             sentiment_sums['POSITIVE'] += prob["score"]
    #         elif prob["label"] == "NEGATIVE":
    #             sentiment_sums['NEGATIVE'] += prob["score"]
    #         else:
    #             sentiment_sums['NEUTRAL'] += prob["score"]

    # # Given that there were actually samples to average over, get the average distro
    # if len(distros) > 0:
    #     avg_sentiment_dist = {label: sentiment_sums[label] / len(distros) for label in sentiment_sums}
    #     total_sum = sum(avg_sentiment_dist.values())
    #     normalized_dist = {label: avg_sentiment_dist[label] / total_sum for label in avg_sentiment_dist}
    #     return normalized_dist
    # return None
    sentiment_sums = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    label_map = {'positive': 'POSITIVE', 'negative': 'NEGATIVE', 'neutral': 'NEUTRAL'}

    for distro in distros:
        for prob in distro:
            label = label_map.get(prob["label"].lower())
            if label:
                sentiment_sums[label] += prob["score"]

    if len(distros) > 0:
        avg_sentiment_dist = {label: sentiment_sums[label] / len(distros) for label in sentiment_sums}
        total_sum = sum(avg_sentiment_dist.values())
        normalized_dist = {label: avg_sentiment_dist[label] / total_sum for label in avg_sentiment_dist}
        return normalized_dist
    return None

def normalize_dict_distro(distro):
    total = sum(distro.values())
    return {label: prob / total for label, prob in distro.items()}

def plot_heatmaps_of_shifts(shift_df):
    sentiment_label_map = {
        'POSITIVE': 'pos_shift',
        'NEUTRAL': 'neutral_shift',
        'NEGATIVE': 'neg_shift'
    }

    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    for idx, sentiment in enumerate(['POSITIVE', 'NEUTRAL', 'NEGATIVE']):
        column_name = sentiment_label_map[sentiment]
        pivot = shift_df.pivot_table(index='in_context_domain', columns='inquiry_domain', values=column_name)
        sns.heatmap(pivot, annot=True, cmap='coolwarm', center=0, linewidths=0.5, ax=axs[idx], cbar_kws={'label': 'Shift'})
        axs[idx].set_title(f'{sentiment} Sentiment Shift')
        axs[idx].set_xlabel('Inquiry Domain Y')
        if idx == 0:
            axs[idx].set_ylabel('In-Context Domain X')
    plt.suptitle('Sentiment Shifts from Control Distribution', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
def plot_heatmaps_of_kl_divergence(kl_div_df):

    kl_rows = []
    for _, row in kl_div_df.iterrows():
        kl_rows.append({
            'In-Context Domain X': row["in_context_domain"],
            'Inquiry Domain Y': row["inquiry_domain"],
            'KL Divergence': row["kl_divergence"]
        })

    df = pd.DataFrame(kl_rows)
    pivot = df.pivot_table(index='In-Context Domain X', columns='Inquiry Domain Y', values='KL Divergence')

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, cmap='Purples', linewidths=0.5, cbar_kws={'label': 'KL Divergence'})
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

def load_jsonl_as_dataframe(cross_domain_run_folder_path):
    data = []
    with open(cross_domain_run_folder_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    df.to_csv("funny_cross_domain_samples.csv", index=False)
    return df

def main(cross_domain_run_folder_path):
    # Collect the sentiment distros for the full cross-domain run 
    cross_domain_run_df = load_jsonl_as_dataframe(cross_domain_run_folder_path)
    # cross_domain_run_df = pd.read_csv("funny_cross_domain_samples.csv")
    all_model_answers = cross_domain_run_df["model_answer"].tolist()
    all_sentiment_distros = get_sentiment_distro_batch(all_model_answers)
    cross_domain_run_df["sentiment_distro"] = all_sentiment_distros
    
    kl_divs_for_all_responses = []
    # For every inquiry subreddit 
    for inquiry_domain, inquiry_group in cross_domain_run_df.groupby("inquiry_domain"):

        # Get the avg control distro (presumably it was run many times), which is when the in-context domain is None
        control_responses_for_inquiry = inquiry_group[inquiry_group["in_context_domain"].isna()]
        aggregated_control_distro = aggregate_dict_distros(control_responses_for_inquiry["sentiment_distro"])
        avg_control_distro = normalize_dict_distro(aggregated_control_distro)

        # For each in-context subreddit it was compared to, across all generated responses 
        for _, row in inquiry_group.iterrows():
            # Disregard control rows
            if pd.isna(row["in_context_domain"]):
                continue
            
            in_context_distro = aggregate_dict_distros([row["sentiment_distro"]])
            kl_for_this_generated_response = calculate_kl_divergence(
                avg_control_distro,
                in_context_distro
            )

            shift = calculate_shift(in_context_distro, avg_control_distro)

            kl_divs_for_all_responses.append({
                "inquiry_domain": inquiry_domain,
                "in_context_domain": row["in_context_domain"],
                "avg_control_distro_for_inquiry": avg_control_distro,
                "pos_shift": shift['POSITIVE'],
                "neg_shift": shift['NEGATIVE'],
                "neutral_shift": shift['NEUTRAL'],
                "kl_divergence": np.sum(kl_for_this_generated_response)
            })

    kl_df = pd.DataFrame(kl_divs_for_all_responses)

    avg_kl_df = kl_df.groupby(
    ["inquiry_domain", "in_context_domain"]
    )["kl_divergence"].mean().reset_index()

    avg_shifts_df = kl_df.groupby(
        ["inquiry_domain", "in_context_domain"]
        )[["pos_shift", "neg_shift", "neutral_shift"]].mean().reset_index()


    print(avg_kl_df)

    # TY chatgpt for helping set up these labeled heatmaps 
    plot_heatmaps_of_kl_divergence(avg_kl_df)
    plot_heatmaps_of_shifts(avg_shifts_df)
if __name__ == "__main__":

    cross_domain_run_folder_path = 'runs/reddit/implicit/reddit_implicit_questions_experiment_2ba223a4b77e4503b527a2ef08c43ce3/results.jsonl'

    main(cross_domain_run_folder_path=cross_domain_run_folder_path)