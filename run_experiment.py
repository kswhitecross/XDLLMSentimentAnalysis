from datasets import books, youtube, news
from experiments.ollama_experiment import BookSummaryWithContextExperiment

# get datasets
books_dataset = books.BooksDataset(file_path="/Users/polinapetrova/data/booksummaries.txt")
youtube_dataset = youtube.YoutubeDataset(file_path="/Users/polinapetrova/data/YoutubeCommentsDataSet.csv")

# run experiment
experiment = BookSummaryWithContextExperiment(model="Llama-3", tokenizer=None, inquiry_dataset=books_dataset, in_context_dataset=youtube_dataset)

for result in experiment._get_experiment_generator():
    print(f"Prompt: {result['prompt']}")
    print(f"Model Answer: {result['model_answer']}")
    print(f"Justification: {result['justification']}")
    print(f"Rating: {result['rating']}")