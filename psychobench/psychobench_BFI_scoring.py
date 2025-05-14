import os
import numpy as np
import matplotlib.pyplot as plt
from utils import get_questionnaire, convert_data, hypothesis_testing, compute_statistics, plot_bar_chart, plot_bar_chart_special


def analyze_shuffled_results_from_subreddit(questionnaire, model, subreddit, testing_file, significance_level =  0.05):
    ''' Taking the initial analysis_results func from the Psychobench repo, and then altering it to fit our subreddit in-context domain comparison needs, comparing to human crowd'''


    # Place to put all the markdown files of tables
    table_of_results_dir =  os.path.join("psychobench", "results", "tables")
   
    # Properly label with the in-context subreddit given
    table_of_results_file = f'{questionnaire["name"]}_{model}_significance={significance_level}_results' if subreddit == None else f'{questionnaire["name"]}_{model}_{subreddit}_significance={significance_level}_results'
    os.makedirs(table_of_results_dir, exist_ok=True)
    table_of_results_path = os.path.join(table_of_results_dir, table_of_results_file)


    # Takes the ugly format from all the shuffling and reorgs back into neat dicts of question:answer pairs per shuffled run
    test_data = convert_data(questionnaire, testing_file)
   
    test_results = compute_statistics(questionnaire, test_data)
       
    cat_list = [cat['cat_name'] for cat in questionnaire['categories']]
    crowd_list = [(c["crowd_name"], c["n"]) for c in questionnaire['categories'][0]["crowd"]]
    mean_list = [[] for i in range(len(crowd_list) + 1)]
   
    output_list = f'# {questionnaire["name"]} Results\n\n'
    output_list += f'| Category | {model} (n = {len(test_data)}) | ' + ' | '.join([f'{c[0]} (n = {c[1]})' for c in crowd_list]) + ' |\n'
    output_list += '| :---: | ' + ' | '.join([":---:" for i in range(len(crowd_list) + 1)]) + ' |\n'
    output_text = ''


    # Analysis by each category
    for cat_index, cat in enumerate(questionnaire['categories']):
        output_text += f'## {cat["cat_name"]}\n'
        output_list += f'| {cat["cat_name"]} | {test_results[cat_index][0]:.1f} $\pm$ {test_results[cat_index][1]:.1f} | '
        mean_list[0].append(test_results[cat_index][0])
       
        for crowd_index, crowd_group in enumerate(crowd_list):
            crowd_data = (cat["crowd"][crowd_index]["mean"], cat["crowd"][crowd_index]["std"], cat["crowd"][crowd_index]["n"])
            result_text, result_list = hypothesis_testing(test_results[cat_index], crowd_data, significance_level, model, crowd_group[0])
            output_list += result_list
            output_text += result_text
            mean_list[crowd_index+1].append(crowd_data[0])
           
        output_list += '\n'
   
    figure_name = table_of_results_file + '_barplot.png'
    figure_title = f"{model}'s Results\n From Answering Big Five Inventory (BFI) Questionnaire\n Given 10 Samples From In-Context Domain: {subreddit}"


    plot_bar_chart(mean_list, cat_list, [model] + [c[0] for c in crowd_list], save_name=figure_name, title=figure_title)
    output_list += f'\n\n![Bar Chart](../figures/{figure_name} "Bar Chart of {model} on {questionnaire["name"]} Given In-Context Domain: {subreddit}")\n\n'
   
    # Writing the results into a text file
    with open(table_of_results_path + '.md', "w", encoding="utf-8") as f:
        f.write(output_list + output_text)


    return test_results


def analyze_shuffled_results_from_subreddit_against_baseline(questionnaire, model, subreddit, testing_file, baseline_results=None, significance_level=0.05):
    ''' Taking the initial analysis_results func from the Psychobench repo, and then altering it to fit our subreddit in-context domain comparison needs, comparing to baseline LLM response '''

    table_of_results_dir = os.path.join("psychobench", "results", "tables")
    table_of_results_file = f'{questionnaire["name"]}_{model}_significance={significance_level}_results' \
        if subreddit is None else f'against_baseline_{questionnaire["name"]}_{model}_{subreddit}_significance={significance_level}_results'
    os.makedirs(table_of_results_dir, exist_ok=True)
    table_of_results_path = os.path.join(table_of_results_dir, table_of_results_file)


    # Load and convert data
    test_data = convert_data(questionnaire, testing_file)
    test_results = compute_statistics(questionnaire, test_data)


    # Labels and formatting
    cat_list = [cat['cat_name'] for cat in questionnaire['categories']]
    crowd_list = [(c["crowd_name"], c["n"]) for c in questionnaire['categories'][0]["crowd"]]
    mean_list = [[] for _ in range(2 if baseline_results else len(crowd_list) + 1)]


    output_list = f'# {questionnaire["name"]} Results\n\n'
   
    output_list += f'| Category | {model} Given In-Context Domain: {subreddit} (n = {len(test_data)}) | '
    output_list += f'{model} Baseline (n = {len(test_data)})' if baseline_results else ' | '.join([f'{c[0]} (n = {c[1]})' for c in crowd_list]) + ' |\n'
    output_list += '| :---: | ' + ' | '.join([":---:" for i in range(len(crowd_list) + 1)]) + ' |\n'
    output_text = ''



    # AI Attribution for ChatGPT:
    # Prompt: Produce the markdown tables and bar plots comparing to the baseline results instead of the human crowd  + the above code + sharing what a single result_list from hypothesis testing looked like 
    # Reflection: It did exactly what I needed, although I had to tweak the titles for barplot and tables and whatnot
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for cat_index, cat in enumerate(questionnaire['categories']):
        output_text += f'## {cat["cat_name"]}\n'
        model_mean, model_std, model_n = test_results[cat_index]
        output_list += f'| {cat["cat_name"]} | {model_mean:.1f} $\pm$ {model_std:.1f} | '
        mean_list[0].append(model_mean)


        if baseline_results:
            base_mean, base_std, base_n = baseline_results[cat_index]
            result_text, result_list = hypothesis_testing(test_results[cat_index], (base_mean, base_std, base_n), significance_level, f'{model} Given In-Context Domain: {subreddit}', f'{model} Baseline')
            output_list += result_list
            output_text += result_text
            mean_list[1].append(base_mean)
        else:
            for crowd_index, crowd_group in enumerate(crowd_list):
                crowd_data = (
                    cat["crowd"][crowd_index]["mean"],
                    cat["crowd"][crowd_index]["std"],
                    cat["crowd"][crowd_index]["n"]
                )
                result_text, result_list = hypothesis_testing(test_results[cat_index], crowd_data, significance_level, model, crowd_group[0])
                output_list += result_list
                output_text += result_text
                mean_list[crowd_index + 1].append(crowd_data[0])


        output_list += '\n'


    figure_name = table_of_results_file + '_barplot.png'
    figure_title = f"{model}'s Results\nFrom Answering {questionnaire['name']} Questionnaire\nGiven 10 Samples From In-Context Domain: {subreddit}"
    label_list = [model] + (["Baseline"] if baseline_results else [c[0] for c in crowd_list])
    plot_bar_chart(mean_list, cat_list, label_list, save_name=figure_name, title=figure_title)


    output_list += f"\n\n![Bar Chart](../figures/{figure_name} \"Bar Chart of {model} on {questionnaire['name']} Given In-Context Domain: {subreddit}\")\n\n"


    with open(table_of_results_path + '.md', "w", encoding="utf-8") as f:
        f.write(output_list + output_text)


    return test_results
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def main():
    subreddits = ["funny", "AskReddit", "gaming", "worldnews", "todayilearned", "Awww", "Music", "memes", "movies", "Showerthoughts", None]
    model = 'Llama-3.1-8B-Instruct'
    questionnaire_name = 'BFI'
    questionnaire = get_questionnaire(questionnaire_name)


    all_results = {}

    # First, get the neat scoring for BFI across all in-context domain experiments like the OG Psychobench analysis against the human crowd, which provides tables in markdown
    for subreddit in subreddits:
        path_to_ugly_result = os.path.join('psychobench', 'results', f'{model}-{questionnaire_name}.csv' if subreddit == None else f'{model}-{questionnaire_name}-{subreddit}.csv')
       
        test_results = analyze_shuffled_results_from_subreddit(questionnaire=questionnaire, model=model, subreddit=subreddit, testing_file=path_to_ugly_result)
        all_results[subreddit or 'baseline'] = test_results

    # Repeat with tables/tests against the baseline/control LLM results instead of human crowd
    for subreddit in subreddits:
        path_to_ugly_result = os.path.join('psychobench', 'results', f'{model}-{questionnaire_name}.csv' if subreddit == None else f'{model}-{questionnaire_name}-{subreddit}.csv')
       
        test_results = analyze_shuffled_results_from_subreddit_against_baseline(questionnaire=questionnaire, model=model, subreddit=subreddit, testing_file=path_to_ugly_result,
                                                                                baseline_results=all_results['baseline'])
        all_results[subreddit or 'baseline'] = test_results

    # Now we want to accumulate all the results, crowd to baseline LLM to each subreddit fed results for the bar plot of all of it 
    cat_list = [cat['cat_name'] for cat in questionnaire['categories']]

    value_list = []
    item_list = []
    human_means = []
    for cat in questionnaire['categories']:
        crowd_group_means = [crowd['mean'] for crowd in cat['crowd']]
        overall_mean = sum(crowd_group_means) / len(crowd_group_means)
        human_means.append(overall_mean)

    value_list.append(human_means)
    item_list.append("Human Crowd")

    for subreddit in subreddits:
        subreddit_key = subreddit if subreddit is not None else 'baseline'
        if all_results[subreddit_key] is None:
            continue
        means = [mean for mean, std, n in all_results[subreddit_key]]
        value_list.append(means)
        item_list.append(subreddit_key if subreddit_key != 'baseline' else '')


    plot_title = f"{questionnaire_name} Results For {model} Given In-Context Domain\n VS {model} Control VS Human Crowd"
    plot_filename = f"{questionnaire_name}_{model}_all_subreddits_barplot.png"
    plot_bar_chart_special(value_list, cat_list, item_list, save_name=plot_filename, title=plot_title)


if __name__ == "__main__":
    main()

