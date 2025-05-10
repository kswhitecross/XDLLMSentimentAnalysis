import os
import json
import numpy as np
from transformers import pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

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


def compute_expected_sentiment_scores_across_subreddit_posts(df):
    ''' Per inquiry_domain subreddit name, 
        get the expected sentiment (avg) per inq_doc_idx within that subreddit
        for both the control case and experimental case. '''  

    # Make sure the scores and indices numeric
    df['score'] = df['score'].astype(float)
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
    # Ensure shift is a number
    df["sentiment_shift"] = df["sentiment_shift"].astype(float)

    summary = (df.groupby(['in_context_domain', 'inquiry_domain']) 
                .agg(expected_sentiment_shift=('sentiment_shift', 'mean'),
                     std_sentiment_shift=('sentiment_shift', 'std'))
                .reset_index())

    return summary


def check_control_vs_control_shift(df):
    # Get the expected sentiment shifts for domain given no context
    control_rows = df[df["in_context_domain"].isna()]

    # Just sanity check the shift is actually 0
    control_shift__not_zero = control_rows[control_rows["sentiment_shift"] == 0]

    if control_shift__not_zero:
        print("There was a shift between control and control so there is a bug!!")
    else:
        print("Control does not differ from control, as we hoped")


def plot_heatmap(df, title, value_to_plot, vmin, vmax, save_file_dir = None, save_file_name = None, show_plot = False, cmap = "coolwarm"):
    # Per https://seaborn.pydata.org/generated/seaborn.heatmap.html, the rows AKA index is the in-context domain
    # Cols are inquiry
    # And we use the expected sentiment shift for this
    pivot_table = df.pivot(index='in_context_domain', columns='inquiry_domain', values=value_to_plot)

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap=cmap, fmt='.4f', 
                vmin=vmin, vmax=vmax, cbar_kws={'label': 'Expected Sentiment Shift'}, center=0.0)
    plt.title(title)
    plt.ylabel("In-Context Domain")
    plt.xlabel("Inquiry Domain")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_file_name != None and save_file_dir != None:
        os.makedirs(save_file_dir, exist_ok=True)
        save_name = os.path.join(save_file_dir, save_file_name)
        print(f"Saving heatmap to: {save_name}")
        plt.savefig(save_name)

    if show_plot:
        plt.show()

def run_sentiment_analysis(sentiment_model, df):
    # Get the col of answers
    model_answers =  list(df['model_answer'])

    # Num answers x 2 for pos, neg
    distro_batch_raw = sentiment_model(model_answers)
    cols_to_add_to_df  = [{distro['label'].lower(): distro['score'] for distro in row} for row in distro_batch_raw]
    sentiment_df = pd.DataFrame(cols_to_add_to_df)

    # In this case, there are more than one scores labels in the distro
    # To start, lets get the shift by adding  the max label as the overall score
    sentiment_columns = ['negative', 'positive']
    sentiment_df['sentiment_label'] = sentiment_df[sentiment_columns].idxmax(axis=1)
    # Then mapping that max label to a number to calc the shift
    label_to_score = {'negative': -1, 'positive': 1}
    sentiment_df['score'] = sentiment_df['sentiment_label'].map(label_to_score)
    return sentiment_df


def get_shifts_and_plot_from_hf_sentiment_distros(sentiment_model, recalculate_hf_distros = False):
                 
   for sentiment_class in ['positive', 'negative', 'binary']:
             
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"FOCUSING ON {sentiment_class} SHIFTS....")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


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
                    print(f"GETTING HF SCORE DISTROS WITH {num_in_context_type}, {model_name}....")
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    llama_outputs =  load_jsonl_as_dataframe(os.path.join(sub_folder, 'results.jsonl'), save_as_csv = False)

                    if recalculate_hf_distros:
                        respective_sentiment_scores = run_sentiment_analysis(sentiment_model, llama_outputs)
                        # Sentiments for this num context type, this model version
                        output_dir = os.path.join("hf_sentiment_from_scratch", num_in_context_type)
                        # If dir already exists that is fine otherwise make it
                        os.makedirs(output_dir, exist_ok=True)
                        # We want the same structure as with sentiment folder from the LLM scores
                        sentiment_save_path = os.path.join(output_dir, f"{model_name}.jsonl")
                        respective_sentiment_scores.to_json(sentiment_save_path, orient='records', lines=True)

                    else:
                        respective_sentiment_scores = load_jsonl_as_dataframe(os.path.join('hf_sentiment_from_scratch', num_in_context_type, model_name + '.jsonl'), save_as_csv = False)

                    # Add the sentiment scores as 
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


                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    print(f"CALCULATING SHIFTS WITH {num_in_context_type}, {model_name}....")
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

                    # Since we are only focusing on one class at a time, we are treating this as the 'score' variable
                    if sentiment_class != 'binary':
                        df_honed_in_on_class_score = llama_combined_with_sentiments.copy()
                        df_honed_in_on_class_score['score'] = df_honed_in_on_class_score[sentiment_class]  
                    else:
                        df_honed_in_on_class_score = llama_combined_with_sentiments.copy()

                    # First get the expected sentiment per post within an inquiry domain
                    expected_sentiment_per_posts = compute_expected_sentiment_scores_across_subreddit_posts(df_honed_in_on_class_score)
                    # Then get the expected sentiment shift per post within an inquiry domain
                    expected_sentiment_shifts_per_posts = compute_sentiment_shift_of_expectations_across_subreddit_posts(expected_sentiment_per_posts)
                    # Lastly get the average of these expected shifts per post, for the expected shift per inquiry domain given in-context domain
                    expected_sentiment_shifts_per_inquiry_domain = compute_expected_sentiment_shift_per_domain(expected_sentiment_shifts_per_posts)
                    # Sanity check that the shift calcs are 0 for control vs control
                    # check_control_vs_control_shift(expected_sentiment_shifts_per_inquiry_domain)

                    # Collect shift values for global vmin/vmax
                    global_shifts.extend(expected_sentiment_shifts_per_inquiry_domain['expected_sentiment_shift'].tolist())
                    # Cache for second pass
                    expected_shifts_cache.append((sentiment_class, num_in_context_type, model_name, expected_sentiment_shifts_per_inquiry_domain))
        
        # This is so that our coloration is consistent (AKA more blue, more neg. More red, more pos)
        # if sentiment_class == "negative":
        #     global_shift_max = -min(global_shifts)
        #     global_shift_min = -max(global_shifts)
        # else:
        global_shift_max = max(global_shifts)
        global_shift_min = min(global_shifts)    
        
        plot_heatmaps_from_cache(global_shift_max=global_shift_max, global_shift_min=global_shift_min, 
                                expected_shifts_cache=expected_shifts_cache, sentiment_source='HF_distilbert',
                                plot_only_standardized = True)


def plot_heatmaps_from_cache(global_shift_min, global_shift_max, expected_shifts_cache, sentiment_source, plot_only_standardized = False):
    for sentiment_class, num_in_context_type, model_name, shift_df in expected_shifts_cache:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"PLOTTING HEATMAP FOR {num_in_context_type}, {model_name}....")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            save_file_dir = os.path.join("heatmaps", sentiment_source, sentiment_class, num_in_context_type, model_name)
            
            # If we are doing an neg shift, we want the more blue to suggest it is becoming increasingly negative 
            if sentiment_class == "negative":
                cmap = "coolwarm_r"
            else:
                cmap = "coolwarm"

            # If we also want to visualize with the stronger coloration, even if not comparable color labels 
            if not plot_only_standardized:
                save_file_name = model_name + "_" + num_in_context_type + '_sentiment_heatmap.png'
                plot_heatmap(df=shift_df,
                            title = f"Expected {sentiment_class.capitalize()} Sentiment Shifts For Inquiry Domains Given In-Context Domains",
                            value_to_plot='expected_sentiment_shift',
                            save_file_dir=save_file_dir,
                            save_file_name=save_file_name,
                            show_plot=False,
                            vmin=None,
                            vmax=None,
                            cmap=cmap)
            
            save_file_name = model_name + "_" + num_in_context_type + '_sentiment_standardized_heatmap.png'
            plot_heatmap(df=shift_df,
                        title = f"Expected {sentiment_class.capitalize()} Sentiment Shifts For Inquiry Domains Given In-Context Domains",
                        value_to_plot='expected_sentiment_shift',
                        save_file_dir=save_file_dir,
                        save_file_name=save_file_name,
                        show_plot=False,
                        vmin=global_shift_min,
                        vmax=global_shift_max,
                        cmap=cmap)
                

        # If we also want to visualize with the stronger coloration, even if not comparable color labels 
            if not plot_only_standardized:
                save_file_name = model_name + "_" + num_in_context_type + '_sentiment_std_heatmap.png'
                plot_heatmap(df=shift_df,
                            title = f"Std of {sentiment_class.capitalize()} Sentiment Shifts For Inquiry Domains Given In-Context Domains",
                            value_to_plot='std_sentiment_shift',
                            save_file_dir=save_file_dir,
                            save_file_name=save_file_name,
                            show_plot=False,
                            vmin=None,
                            vmax=None,
                            cmap=cmap)
            
            save_file_name = model_name + "_" + num_in_context_type + '_sentiment_standardized_std_heatmap.png'
            plot_heatmap(df=shift_df,
                        title = f"Std of {sentiment_class.capitalize()} Sentiment Shifts For Inquiry Domains Given In-Context Domains",
                        value_to_plot='std_sentiment_shift',
                        save_file_dir=save_file_dir,
                        save_file_name=save_file_name,
                        show_plot=False,
                        vmin=global_shift_min,
                        vmax=global_shift_max,
                        cmap=cmap)

def main():
    #  distilbert/distilbert-base-uncased-finetuned-sst-2-english
    sentiment_model = pipeline("sentiment-analysis", truncation=True, top_k=None)
    get_shifts_and_plot_from_hf_sentiment_distros(sentiment_model, recalculate_hf_distros = False)

if __name__ == "__main__":
    main()














    
# def get_shifts_and_plot_from_llm_sentiments():
#     # Collect this for consistent color scale
#     global_shifts = []  

#     # So ya don't need to recalculate everything
#     expected_shifts_cache = [] 

#     for num_in_context_type in ['long', 'original']:
          
#         print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#         print(f"ANALYZING LLAMA MODELS with {num_in_context_type} IN-CONTEXT SAMPLES....")
#         print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

#         folder = Path(num_in_context_type)
#         for sub_folder in folder.iterdir():
#             if sub_folder.is_dir():
                      

#                 # Get the model name of the current directory of results
#                 sub_folder_name_parts = sub_folder.name.split("_")
#                 # if sub_folder_name_parts[1] == '70B':
#                 #     model_name = "_".join(sub_folder_name_parts[:3])
#                 # else:
#                 #     model_name = "_".join(sub_folder_name_parts[:2])
#                 model_name = "_".join(sub_folder_name_parts[:2])

            
#                 print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#                 print(f"CALCULATING SHIFTS WITH {num_in_context_type}, {model_name}....")
#                 print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


#                 llama_outputs =  load_jsonl_as_dataframe(os.path.join(sub_folder, 'results.jsonl'), save_as_csv = False)
#                 respective_sentiment_scores = load_jsonl_as_dataframe(os.path.join('sentiment', num_in_context_type, model_name + '.jsonl'), save_as_csv = False)

#                 # Add the sentiment scores as cols
#                 llama_combined_with_sentiments =  pd.concat([llama_outputs, respective_sentiment_scores], axis=1)
                
#                 # Just a sanity check that our scores in fact do align fully with the OG output rows
#                 print("Llama outputs shape:", llama_outputs.shape)
#                 print("Sentiment scores shape:", respective_sentiment_scores.shape)
#                 print("Merged shape:", llama_combined_with_sentiments.shape)
#                 print("# Null doc idx:", llama_combined_with_sentiments['inq_doc_idx'].isna().sum())
#                 print("# Null scores:", llama_combined_with_sentiments['score'].isna().sum())
#                 print("The NaN rows, if any:", respective_sentiment_scores[respective_sentiment_scores['score'].isna()])

#                 # TODO check in on this
#                 llama_combined_with_sentiments = llama_combined_with_sentiments[llama_combined_with_sentiments['score'].notna()]

#                 # First get the expected sentiment per post within an inquiry domain
#                 expected_sentiment_per_posts = compute_expected_sentiment_scores_across_subreddit_posts(llama_combined_with_sentiments)
#                 # Then get the expected sentiment shift per post within an inquiry domain
#                 expected_sentiment_shifts_per_posts = compute_sentiment_shift_of_expectations_across_subreddit_posts(expected_sentiment_per_posts)
#                 # Lastly get the average of these expected shifts per post, for the expected shift per inquiry domain given in-context domain
#                 expected_sentiment_shifts_per_inquiry_domain = compute_expected_sentiment_shift_per_domain(expected_sentiment_shifts_per_posts)
#                 # Sanity check that the shift calcs are 0 for control vs control
#                 # check_control_vs_control_shift(expected_sentiment_shifts_per_inquiry_domain)

#                 # Collect shift values for global vmin/vmax
#                 global_shifts.extend(expected_sentiment_shifts_per_inquiry_domain['expected_sentiment_shift'].tolist())
#                 # Cache for second pass
#                 expected_shifts_cache.append((num_in_context_type, model_name, expected_sentiment_shifts_per_inquiry_domain))
    
#     global_shift_max = max(global_shifts)
#     global_shift_min = min(global_shifts)

    
#     plot_heatmaps_from_cache(global_shift_max=global_shift_max, global_shift_min=global_shift_min, 
#                              expected_shifts_cache=expected_shifts_cache, sentiment_source='LLM_scoring',
#                              plot_only_standardized = True)
