import os
import json
import numpy as np
from transformers import pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def load_jsonl_as_dataframe(cross_domain_run_folder_path, save_as_csv = False):

    data = []
    with open(cross_domain_run_folder_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)

    if save_as_csv:
        file_path = Path(cross_domain_run_folder_path)
        df.to_csv(file_path.stem, index=False)
    print(df.head())

    return df


def expected_sentiment_scores_across_subreddit_posts(df):
    ''' Per inquiry_domain subreddit name, 
        get the expected sentiment (avg) per inq_doc_idx within that subreddit
        for both the control case and experimental case. '''  

    # Make sure the scores and indices numeric
    df['score'] = df['score'].astype(int)
    df['inq_doc_idx'] = df['inq_doc_idx'].astype(int)

    expected_score_per_post_within_domain_given_context = (
    # Get the average sentiment score for this post within this cross-domain analysis 
    # (including the control when in_context is None, so dont have Pandas drop the rows with NaN)
    df.groupby(["in_context_domain", "inquiry_domain", "inq_doc_idx"], dropna=False)["score"]
    .mean()
    # go from the series that auto has score as the val name, reset indexing to be from 0 in order back as a data frame
    .reset_index()
    # Now it is the Expected Score, not just Score
    .rename(columns={"score": "expected_score"})
    )
    return expected_score_per_post_within_domain_given_context


def compute_sentiment_shift_of_expectations_across_subreddit_posts(df):
    # We want to find the control scores per doc within an inquiry domain subreddit
    control_scores = df[df["in_context_domain"].isna()][["inquiry_domain", "inq_doc_idx", "expected_score"]]
    control_scores = control_scores.rename(columns={"expected_score": "control_score"})

    # Join the control scores on each doc within an inquiry domain subreddit
    df_with_control = pd.merge(df, control_scores, on=["inquiry_domain", "inq_doc_idx"], how="left")

    # Now we can just directly add a shift col of the diff for that doc within each subreddit
    df_with_control["sentiment_shift"] = df_with_control["expected_score"] - df_with_control["control_score"]

    return df_with_control


def compute_expected_sentiment_shift_per_domain(df):
    # Ensure these are all ints 
    df["sentiment_shift"] =  df["sentiment_shift"].astype(int)

    # Now, for every in_context - inquiry domain pairing, get the expected shift from the control 
    expected_shift_per_inquiry_given_context = df.groupby(['in_context_domain', 'inquiry_domain'])['sentiment_shift'].mean().reset_index()
    expected_shift_per_inquiry_given_context.rename(columns={'sentiment_shift': 'expected_sentiment_shift'}, inplace=True)

    return expected_shift_per_inquiry_given_context


def check_control_vs_control_shift(df):
    # Get the expected sentiment shifts for domain given no context
    control_rows = df[df["in_context_domain"].isna()]

    # Just sanity check the shift is actually 0
    control_shift__not_zero = control_rows[control_rows["sentiment_shift"] == 0]

    if control_shift__not_zero:
        print("There was a shift between control and control so there is a bug!!")
    else:
        print("Control does not differ from control, as we hoped")


def plot_heatmap(df):
    # Per https://seaborn.pydata.org/generated/seaborn.heatmap.html, the rows AKA index is the in-context domain
    # Cols are inquiry
    # And we use the expected sentiment shift for this
    pivot_table = df.pivot(index='in_context_domain', columns='inquiry_domain', values='expected_sentiment_shift')

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt='.4f', cbar_kws={'label': 'Expected Sentiment Shift'})
    plt.title("Expected Sentiment Shifts For Inquiry Domains Given In-Context Domains")
    plt.ylabel("In-Context Domain")
    plt.xlabel("Inquiry Domain")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def main():
    og_llama_70b_outputs =  load_jsonl_as_dataframe('llama_70b_q/llama_70B_Q_13b9cb55548f4b3b889b572a58aa7815/results.jsonl', save_as_csv = False)
    respective_sentiment_scores = load_jsonl_as_dataframe('sentiment/sentiment/original/llama_70b.jsonl', save_as_csv = False)

    # Add the sentiment scores as cols
    llama_70b_combined_with_sentiments =  pd.concat([og_llama_70b_outputs, respective_sentiment_scores], axis=1)
    
    print("Original outputs shape:", og_llama_70b_outputs.shape)
    print("Sentiment scores shape:", respective_sentiment_scores.shape)
    print("Merged shape:", llama_70b_combined_with_sentiments.shape)
    print(llama_70b_combined_with_sentiments['inq_doc_idx'].isna().sum())

    expected_sentiment_per_posts = expected_sentiment_scores_across_subreddit_posts(llama_70b_combined_with_sentiments)
    expected_sentiment_shifts_per_posts = compute_sentiment_shift_of_expectations_across_subreddit_posts(expected_sentiment_per_posts)
    expected_sentiment_shifts_per_inquiry_domain = compute_expected_sentiment_shift_per_domain(expected_sentiment_shifts_per_posts)
    # check_control_vs_control_shift(expected_sentiment_shifts_per_inquiry_domain)

    plot_heatmap(expected_sentiment_shifts_per_inquiry_domain)
    
if __name__ == "__main__":
    main()