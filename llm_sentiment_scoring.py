import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import pipeline

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


def compute_std_sentiment_scores_across_subreddit_posts(df):
    ''' Per inquiry_domain subreddit name, 
        get the std of sentiment per inq_doc_idx within that subreddit
        for both the control case and experimental case. '''  

    # Make sure the scores and indices are numeric
    df['score'] = df['score'].astype(float)
    df['inq_doc_idx'] = df['inq_doc_idx'].astype(int)

    std_score_per_post_within_domain_given_context = (
        df.groupby(["in_context_domain", "inquiry_domain", "inq_doc_idx"], dropna=False)["score"]
        .std()
        .reset_index()
        .rename(columns={"score": "std_score"})
    )

    return std_score_per_post_within_domain_given_context

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

    summary = (df.groupby(['in_context_domain', 'inquiry_domain'], dropna=False) 
                .agg(expected_sentiment_shift=('sentiment_shift', 'mean'),
                     std_sentiment_shift=('sentiment_shift', 'std'))
                .reset_index())

    return summary


def check_control_vs_control_shift(df):
    ''' Simple unit test where the shift between control and control should be 0 '''
    # Get the expected sentiment shifts for domain given no context
    control_rows = df[df["in_context_domain"].isna()]

    # Just sanity check the shift is actually 0
    control_shift_not_zero = control_rows[control_rows["expected_sentiment_shift"] != 0]

    if control_shift_not_zero.empty == False:
        print("There was a shift between control and control so there is a bug X")
    else:
        print("Control does not differ from control ✔")


def plot_heatmap(df, title, value_to_plot, vmin, vmax, save_file_dir = None, save_file_name = None, show_plot = False, cmap = "coolwarm"):
    # Per https://seaborn.pydata.org/generated/seaborn.heatmap.html, the rows AKA index is the in-context domain
    # Cols are inquiry
    # And we use the expected sentiment shift for this
    df = df[df['in_context_domain'].notna()]

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
    sentiment_columns = sentiment_df.columns
    sentiment_df['sentiment_label'] = sentiment_df[sentiment_columns].idxmax(axis=1)
    # Optional: Map labels to a score
    label_to_score = {'negative': -1, 'neutral': 0, 'positive': 1}
    sentiment_df['score'] = sentiment_df['sentiment_label'].map(label_to_score)

    return sentiment_df


def map_llm_scores(df):
    def map_score(val):
        if pd.isna(val):
            return None
        val = int(val)
        if val <= 2:
            return -1
        elif val == 3:
            return 0
        else:  # 4 or 5
            return 1

    df['score'] = df['score'].apply(map_score)
    return df


def plot_std_sentiment_per_posts(std_sentiment_per_posts, save_dir):
    grouped = std_sentiment_per_posts.groupby(["in_context_domain", "inquiry_domain"])

    print(f"Saving std histograms under: {save_dir}")

    for (context, inquiry), group in grouped:
        plt.figure(figsize=(6, 4))
        sns.histplot(group["std_score"], kde=False)
        plt.title(f"Std Dev of Sentiment Scores\nContext: {context}, Inquiry: {inquiry}")
        plt.xlabel("Standard Deviation")
        plt.ylabel("Number of Posts")
        plt.tight_layout()
        # plt.show()
        save_file_name = f'in_context_{context}_inquiry_{inquiry}.png'
        if save_file_name != None and save_dir != None:
                os.makedirs(save_dir, exist_ok=True)
                save_name = os.path.join(save_dir, save_file_name)
                plt.savefig(save_name)
        

def plot_histogram_of_expected_sentiment_per_posts(expected_sentiment_per_posts, save_dir):
    grouped = expected_sentiment_per_posts.groupby(["in_context_domain", "inquiry_domain"])

    print(f"Saving expectation histograms under: {save_dir}")

    for (context, inquiry), group in grouped:
        plt.figure(figsize=(6, 4))
        sns.histplot(group["expected_score"], kde=True)
        plt.title(f"Expected Sentiment Scores\nContext: {context}, Inquiry: {inquiry}")
        plt.xlabel("Average Sentiment Score")
        plt.ylabel("Number of Posts")
        plt.tight_layout()

        save_file_name = f'in_context_{context}_inquiry_{inquiry}_mean.png'
        if save_file_name and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, save_file_name)
            plt.savefig(save_path)



def plot_histogram_of_expected_sentiment_shifts(expected_sentiment_shifts_per_posts, save_dir):
    grouped = expected_sentiment_shifts_per_posts.groupby(["in_context_domain", "inquiry_domain"])

    print(f"Saving shift expectation histograms under: {save_dir}")

    for (context, inquiry), group in grouped:
        plt.figure(figsize=(6, 4))
        sns.histplot(group["sentiment_shift"], kde=True)
        plt.title(f"Expected Sentiment Scores Shifts\nContext: {context}, Inquiry: {inquiry}")
        plt.xlabel("Sentiment Score Shifts")
        plt.ylabel("Number of Posts")
        plt.tight_layout()

        save_file_name = f'in_context_{context}_inquiry_{inquiry}_mean.png'
        if save_file_name and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, save_file_name)
            plt.savefig(save_path)

import os
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

def _plot_single_shift_histogram(args):
    (context, inquiry), group, save_dir = args

    plt.figure(figsize=(6, 4))
    sns.histplot(group["sentiment_shift"], kde=True)
    plt.title(f"Expected Sentiment Score Shifts\nContext: {context}, Inquiry: {inquiry}")
    plt.xlabel("Sentiment Score Shifts")
    plt.ylabel("Number of Posts")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_file_name = f'in_context_{context}_inquiry_{inquiry}_mean.png'
    save_path = os.path.join(save_dir, save_file_name)
    plt.savefig(save_path)
    plt.close()  # Important to avoid memory buildup in multiprocessing

def plot_histogram_of_expected_sentiment_shifts_parallel(expected_sentiment_shifts_per_posts, save_dir, max_workers=4):
    grouped = expected_sentiment_shifts_per_posts.groupby(["in_context_domain", "inquiry_domain"])
    print(f"Saving shift expectation histograms under: {save_dir}")

    tasks = [((context, inquiry), group, save_dir) for (context, inquiry), group in grouped]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(_plot_single_shift_histogram, tasks)


def get_shifts_from_hf_sentiment_distros(include_neutral = True, recalculate_hf_distros = False, plot_shift_histograms = False):
    if include_neutral:
        sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", truncation=True, max_length= 512, top_k=None)
        sentiment_classes =  ['positive', 'negative', 'neutral', 'binary']
    else:
        sentiment_model = pipeline("sentiment-analysis", truncation=True, top_k=None)
        sentiment_classes =  ['positive', 'negative', 'binary'] 

    sentiment_source = 'cardiff' if include_neutral else 'HF_distilbert'

    # Collect this for consistent color scale
    global_shifts = []  

    # So ya don't need to recalculate everything
    expected_shifts_cache = []       
    for sentiment_class in sentiment_classes:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"FOCUSING ON {sentiment_class} SHIFTS....")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
 

        recalculated  = False
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

                    if recalculated == False and recalculate_hf_distros:
                        respective_sentiment_scores = run_sentiment_analysis(sentiment_model, llama_outputs)
                        # Sentiments for this num context type, this model version
                        output_dir = os.path.join('hf_includes_neutral_sentiment_from_scratch' if include_neutral else 'hf_sentiment_from_scratch', num_in_context_type)
                        # If dir already exists that is fine otherwise make it
                        os.makedirs(output_dir, exist_ok=True)
                        # We want the same structure as with sentiment folder from the LLM scores
                        sentiment_save_path = os.path.join(output_dir, f"{model_name}.jsonl")
                        respective_sentiment_scores.to_json(sentiment_save_path, orient='records', lines=True)

                        # Don't need to recalculate again, done once
                        recalculated = True

                    else:
                        respective_sentiment_scores = load_jsonl_as_dataframe(os.path.join('hf_includes_neutral_sentiment_from_scratch' if include_neutral else 'hf_sentiment_from_scratch', num_in_context_type, model_name + '.jsonl'), save_as_csv = False)

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
                    check_control_vs_control_shift(expected_sentiment_shifts_per_inquiry_domain)

                    # Collect shift values for global vmin/vmax
                    global_shifts.extend(expected_sentiment_shifts_per_inquiry_domain['expected_sentiment_shift'].tolist())
                    # Cache for second pass
                    expected_shifts_cache.append((sentiment_class, num_in_context_type, model_name, expected_sentiment_shifts_per_inquiry_domain))

                    if plot_shift_histograms:
                        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                        print(f"PLOTTING HISTOGRAMS OF POST-LEVEL SENTIMENT SHIFTS PER CROSS-DOMAIN PAIR WITH {num_in_context_type}, {model_name}....")
                        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                        
                        save_file_dir = os.path.join("shift_histograms", sentiment_source, sentiment_class, num_in_context_type, model_name)
                        plot_histogram_of_expected_sentiment_shifts_parallel(expected_sentiment_shifts_per_posts=expected_sentiment_shifts_per_posts, save_dir = save_file_dir)


    # After we have global max/min shifts for this sentiment class
    global_shift_max = max(global_shifts)
    global_shift_min = min(global_shifts)    
    
    return global_shift_max, global_shift_min, sentiment_source, expected_shifts_cache

    
def get_shifts_from_llm_sentiment_scores(plot_shift_histograms = False): 
    sentiment_source = 'llm_scores'

    sentiment_classes =  ['binary', '5_star']

    # Collect this for consistent color scale
    global_shifts = []  

    # So ya don't need to recalculate everything
    expected_shifts_cache = []    
    
    for sentiment_class in sentiment_classes:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"FOCUSING ON {sentiment_class} SHIFTS....")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")    

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
                    print(f"GETTING LLM SCORE DISTROS WITH {num_in_context_type}, {model_name}....")
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    llama_outputs =  load_jsonl_as_dataframe(os.path.join(sub_folder, 'results.jsonl'), save_as_csv = False)
                    # Map these to binary -1 0 1
                    respective_sentiment_scores = load_jsonl_as_dataframe(os.path.join('sentiment', num_in_context_type, model_name + '.jsonl'), save_as_csv = False)
                    if sentiment_class == "binary":
                        respective_sentiment_scores = map_llm_scores(respective_sentiment_scores)

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
                    print(f"CALCULATING EXPECTED SHIFTS WITH {num_in_context_type}, {model_name}....")
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

                    # First get the expected sentiment per post within an inquiry domain
                    expected_sentiment_per_posts = compute_expected_sentiment_scores_across_subreddit_posts(llama_combined_with_sentiments)
                    # Then get the expected sentiment shift per post within an inquiry domain
                    expected_sentiment_shifts_per_posts = compute_sentiment_shift_of_expectations_across_subreddit_posts(expected_sentiment_per_posts)
                    # Lastly get the average of these expected shifts per post, for the expected shift per inquiry domain given in-context domain
                    expected_sentiment_shifts_per_inquiry_domain = compute_expected_sentiment_shift_per_domain(expected_sentiment_shifts_per_posts)
                    # Sanity check that the shift calcs are 0 for control vs control
                    check_control_vs_control_shift(expected_sentiment_shifts_per_inquiry_domain)

                    # Collect shift values for global vmin/vmax
                    global_shifts.extend(expected_sentiment_shifts_per_inquiry_domain['expected_sentiment_shift'].tolist())
                    # Cache for second pass
                    expected_shifts_cache.append((sentiment_class, num_in_context_type, model_name, expected_sentiment_shifts_per_inquiry_domain))
        
                    if plot_shift_histograms:
                        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                        print(f"PLOTTING HISTOGRAMS OF POST-LEVEL SENTIMENT SHIFTS PER CROSS-DOMAIN PAIR WITH {num_in_context_type}, {model_name}....")
                        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                        
                        save_file_dir = os.path.join("shift_histograms", sentiment_source, sentiment_class, num_in_context_type, model_name)
                        plot_histogram_of_expected_sentiment_shifts_parallel(expected_sentiment_shifts_per_posts=expected_sentiment_shifts_per_posts, save_dir = save_file_dir)


    # After we have global max/min shifts for this sentiment class
    global_shift_max = max(global_shifts)
    global_shift_min = min(global_shifts)    
    
    return global_shift_max, global_shift_min, sentiment_source, expected_shifts_cache


def plot_heatmaps_from_cache(global_shift_min, global_shift_max, expected_shifts_cache, sentiment_source = None, plot_only_standardized = False):
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


def plot_heatmaps_from_cache(heatmap_parent_folder_name, global_shift_min, global_shift_max, expected_shifts_cache, sentiment_source, plot_only_standardized = False):
    for sentiment_class, num_in_context_type, model_name, shift_df in expected_shifts_cache:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"PLOTTING HEATMAP FOR {num_in_context_type}, {model_name}....")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            save_file_dir = os.path.join(heatmap_parent_folder_name, sentiment_source, sentiment_class, num_in_context_type, model_name)
            
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

    global_shift_max_hf, global_shift_min_hf, sentiment_source_hf, expected_shifts_cache_hf = get_shifts_from_hf_sentiment_distros(include_neutral=False, recalculate_hf_distros = False)
    global_shift_max_hf_with_neutral, global_shift_min_hf_with_neutral, sentiment_source_hf_with_neutral, expected_shifts_cache_hf_with_neutral = get_shifts_from_hf_sentiment_distros(include_neutral=True, recalculate_hf_distros = False)
    global_shift_max_llm, global_shift_min_llm, sentiment_source_llm, expected_shifts_cache_llm = get_shifts_from_llm_sentiment_scores(include_neutral=True)

    global_shift_max = max(global_shift_max_hf, global_shift_max_hf_with_neutral, global_shift_max_llm)
    global_shift_min = min(global_shift_min_hf, global_shift_min_hf_with_neutral, global_shift_min_llm)

    
    # We can plot all the shifts per combo of num examples type and model size for said class GLOBAL GLOBAL
    plot_heatmaps_from_cache(global_shift_max=global_shift_max, global_shift_min=global_shift_min, 
                            expected_shifts_cache=expected_shifts_cache_hf, sentiment_source=sentiment_source_hf,
                            plot_only_standardized = False, heatmap_parent_folder_name = "heatmaps_global_across_all")
    
    plot_heatmaps_from_cache(global_shift_max=global_shift_max, global_shift_min=global_shift_min, 
                            expected_shifts_cache=expected_shifts_cache_hf_with_neutral, sentiment_source=sentiment_source_hf_with_neutral,
                            plot_only_standardized = False, heatmap_parent_folder_name = "heatmaps_global_across_all")

    
    plot_heatmaps_from_cache(global_shift_max=global_shift_max, global_shift_min=global_shift_min, 
                            expected_shifts_cache=expected_shifts_cache_llm, sentiment_source=sentiment_source_llm,
                            plot_only_standardized = False, heatmap_parent_folder_name = "heatmaps_global_across_all")
        

    # We can plot all the shifts per combo of num examples type and model size for said class GLOBAL TO THE MODEL
    plot_heatmaps_from_cache(global_shift_max=global_shift_max_hf, global_shift_min=global_shift_min_hf, 
                            expected_shifts_cache=expected_shifts_cache_hf, sentiment_source=sentiment_source_hf,
                            plot_only_standardized = False, heatmap_parent_folder_name = "heatmaps_global_to_sentiment_model")
    
    plot_heatmaps_from_cache(global_shift_max=global_shift_max_hf_with_neutral, global_shift_min=global_shift_min_hf_with_neutral, 
                            expected_shifts_cache=expected_shifts_cache_hf_with_neutral, sentiment_source=sentiment_source_hf_with_neutral,
                            plot_only_standardized = False, heatmap_parent_folder_name = "heatmaps_global_to_sentiment_model")

    
    plot_heatmaps_from_cache(global_shift_max=global_shift_max_llm, global_shift_min=global_shift_min_llm, 
                            expected_shifts_cache=expected_shifts_cache_llm, sentiment_source=sentiment_source_llm,
                            plot_only_standardized = False, heatmap_parent_folder_name = "heatmaps_global_to_sentiment_model")
if __name__ == "__main__":
    main()











