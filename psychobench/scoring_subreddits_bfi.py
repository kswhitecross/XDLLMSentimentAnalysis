import os 
import sys 
import csv
from utils import get_questionnaire, convert_data, hypothesis_testing, analysis_personality, compute_statistics, plot_bar_chart

def analyze_shuffled_results_from_subreddit(questionnaire, model, subreddit, testing_file, significance_level =  0.01):
    # Place to put all the markdown files of tables 
    table_of_results_dir =  os.path.join("psychobench", "results", "tables")
    
    # Properly label with the in-context subreddit given
    table_of_results_file = f'{questionnaire["name"]}_{model}_{subreddit}_significance={significance_level}_results'
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



def main():
    subreddits = ["funny", "AskReddit", "gaming", "worldnews", "todayilearned", "Awww", "Music", "memes", "movies", "Showerthoughts"]
    model = 'Llama-3.1-8B-Instruct'
    questionnaire_name = 'BFI'
    questionnaire = get_questionnaire(questionnaire_name)

    for subreddit in subreddits:
        path_to_ugly_result = os.path.join('psychobench', 'results', f'{model}-{questionnaire_name}-{subreddit}.csv') 
        analyze_shuffled_results_from_subreddit(questionnaire=questionnaire, model=model, subreddit=subreddit, testing_file=path_to_ugly_result)

if __name__ == "__main__":
    main()