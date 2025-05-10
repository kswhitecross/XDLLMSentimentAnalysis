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


def plot_heatmap(df, vmin, vmax, save_file_dir = None, save_file_name = None, show = False):
    # Per https://seaborn.pydata.org/generated/seaborn.heatmap.html, the rows AKA index is the in-context domain
    # Cols are inquiry
    # And we use the expected sentiment shift for this
    pivot_table = df.pivot(index='in_context_domain', columns='inquiry_domain', values='expected_sentiment_shift')

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt='.4f', 
                vmin=vmin, vmax=vmax, cbar_kws={'label': 'Expected Sentiment Shift'})
    plt.title("Expected Sentiment Shifts For Inquiry Domains Given In-Context Domains")
    plt.ylabel("In-Context Domain")
    plt.xlabel("Inquiry Domain")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_file_name != None and save_file_dir != None:
        plt.savefig(os.path.join(save_file_dir, save_file_name))

    if show:
        plt.show()

def main():
    # Collect this for consistent color scale
    global_shifts = []  

    # So ya don't need to recalculate everything
    expected_shifts_cache = [] 

    for num_in_context_type in ['long', 'original']:
          
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"ANALYZING LLAMA MODELS with {num_in_context_type} IN-CONTEXT SAMPLES....")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        folder = Path(num_in_context_type)
        for sub_folder in folder.iterdir():
            if sub_folder.is_dir():
                      

                # Get the model name of the current directory of results
                sub_folder_name_parts = sub_folder.name.split("_")
                # if sub_folder_name_parts[1] == '70B':
                #     model_name = "_".join(sub_folder_name_parts[:3])
                # else:
                #     model_name = "_".join(sub_folder_name_parts[:2])
                model_name = "_".join(sub_folder_name_parts[:2])

            
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print(f"CALCULATING SHIFTS WITH {num_in_context_type}, {model_name}....")
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


                llama_outputs =  load_jsonl_as_dataframe(os.path.join(sub_folder, 'results.jsonl'), save_as_csv = False)
                respective_sentiment_scores = load_jsonl_as_dataframe(os.path.join('sentiment', num_in_context_type, model_name + '.jsonl'), save_as_csv = False)

                # Add the sentiment scores as cols
                llama_combined_with_sentiments =  pd.concat([llama_outputs, respective_sentiment_scores], axis=1)
                
                # Just a sanity check that our scores in fact do align fully with the OG output rows
                print("Llama outputs shape:", llama_outputs.shape)
                print("Sentiment scores shape:", respective_sentiment_scores.shape)
                print("Merged shape:", llama_combined_with_sentiments.shape)
                print("# Null doc idx:", llama_combined_with_sentiments['inq_doc_idx'].isna().sum())
                print("# Null scores:", llama_combined_with_sentiments['score'].isna().sum())
                print("The NaN rows, if any:", respective_sentiment_scores[respective_sentiment_scores['score'].isna()])

                # TODO check in on this
                llama_combined_with_sentiments = llama_combined_with_sentiments[llama_combined_with_sentiments['score'].notna()]

                # First get the expected sentiment per post within an inquiry domain
                expected_sentiment_per_posts = expected_sentiment_scores_across_subreddit_posts(llama_combined_with_sentiments)
                # Then get the expected sentiment shift per post within an inquiry domain
                expected_sentiment_shifts_per_posts = compute_sentiment_shift_of_expectations_across_subreddit_posts(expected_sentiment_per_posts)
                # Lastly get the average of these expected shifts per post, for the expected shift per inquiry domain given in-context domain
                expected_sentiment_shifts_per_inquiry_domain = compute_expected_sentiment_shift_per_domain(expected_sentiment_shifts_per_posts)
                # Sanity check that the shift calcs are 0 for control vs control
                # check_control_vs_control_shift(expected_sentiment_shifts_per_inquiry_domain)

                # Collect shift values for global vmin/vmax
                global_shifts.extend(expected_sentiment_shifts_per_inquiry_domain['expected_sentiment_shift'].tolist())
                # Cache for second pass
                expected_shifts_cache.append((num_in_context_type, model_name, expected_sentiment_shifts_per_inquiry_domain))
    
    global_shift_max = max(global_shifts)
    global_shift_min = min(global_shifts)

    for num_in_context_type, model_name, shift_df in expected_shifts_cache:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"PLOTTING HEATMAP FOR {num_in_context_type}, {model_name}....")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            
            save_file_name = model_name + "_" + num_in_context_type + '_heatmap.png'
            plot_heatmap(df=shift_df,
                        save_file_dir=num_in_context_type,
                        save_file_name=save_file_name,
                        show=False,
                        vmin=None,
                        vmax=None)
            
            save_file_name = model_name + "_" + num_in_context_type + '_standardized_heatmap.png'
            plot_heatmap(df=shift_df,
                        save_file_dir=num_in_context_type,
                        save_file_name=save_file_name,
                        show=False,
                        vmin=global_shift_min,
                        vmax=global_shift_max)
        
if __name__ == "__main__":
    main()