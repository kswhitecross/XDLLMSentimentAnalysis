"""
This file contains the Experiment Base class, which defines the methods an experiment must have.
"""
from abc import ABC, abstractmethod
from typing import Generator, Any
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns

class Experiment(ABC):
    """
    Experiment base class.
    """
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name

    @abstractmethod
    def _get_experiment_generator(self) -> Generator[dict[str, Any], None, None]:
        """
        This method creates the experiment generator, which yields dictionaries containing all the relevant information
         to be logged and passed to the model.
        """
        pass

    def create_experiment(self) -> Generator[dict[str, Any], None, None]:
        # create the experiment
        exp_gen = self._get_experiment_generator()

        # create a new generator that wraps it, and verifies it has the necessary keys
        def wrapped_experiment():
            for exp in exp_gen:
                # make sure the experiment dict generated has the keys we need
                assert 'context' in exp
                assert 'max_gen_tokens' in exp
                yield exp
        return wrapped_experiment()
    

    def get_sentiment_from_pretrained_model(self, model_answer: str, sentiment_model_path: str):
        # sentiment_model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        # Load the tokenizer and config for the pretrained sentiment analysis model
        tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path, device_map = 'auto')
        config = AutoConfig.from_pretrained(sentiment_model_path)

        # Load the pretrained model into PyTorch and encode the input text as numbers for the model to understand
        model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path,  device_map = 'auto')
        encoded_input = tokenizer(model_answer, return_tensors='pt')

        # Outputs the raw scores
        output = model(**encoded_input)
        # Turn raw scores into probabilities that sum to 1
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        # Print labels and scores, ranked
        # ranking = np.argsort(scores)
        # ranking = ranking[::-1]
        # for i in range(scores.shape[0]):
        #     l = config.id2label[ranking[i]]
        #     s = scores[ranking[i]]
        #     print(f"{i+1}) {l} {np.round(float(s), 4)}")

        return scores 

    def plot_mean_sentiment_scores(scores_across_samples):
        if len(scores_across_samples) >= 1:
            labels = ['Negative', 'Neutral', 'Positive']
            mean_scores = np.mean(scores_across_samples)
            plt.bar(labels, mean_scores, color=['red', 'blue', 'green'])
            plt.xlabel('Sentiment')
            plt.ylabel('Mean Sentiment Score')
            plt.title('Mean Sentiment Scores Across Samples')
            plt.show()
        else:
            print("No sampled scores provided to plot mean sentiment scores")

    
    def plot_sentiment_score_distribution_across_all_3(scores_across_samples):
        # Flips it on its side
        neg_scores, neutral_scores, pos_scores = np.array(scores_across_samples).T

        plt.figure(figsize=(10, 5))
        sns.histplot(neg_scores, color='red', bins=20, label='Negative', kde=True, alpha=0.5)
        sns.histplot(neutral_scores, color='blue', bins=20,label='Neutral', kde=True, alpha=0.5)
        sns.histplot(pos_scores, color='green', bins=20,label='Positive', kde=True, alpha=0.5)

        plt.xlabel("Sentiment Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.title("Distribution of Sentiment Scores")
        plt.show()

    
    def plot_predicted_sentiment_counts(scores_across_samples):
        labels = ['Negative', 'Neutral', 'Positive']

        predicted_labels = []
        for score in scores_across_samples:
            predicted_index = np.argmax(score)
            predicted_labels.append(labels[predicted_index])

        plt.figure(figsize=(8, 6))
        sns.countplot(x=predicted_labels, palette='coolwarm')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.title('Distribution of Predicted Sentiments')
        plt.show()

    
    @abstractmethod
    def evaluate_results(self, result_dict: dict[str, Any]):
        """
        This method gets the result dict containing the generated output `model_answer`, evaluates it, and adds the
         evaluation results to be logged to `result_dict`
        """
        pass

    @property
    @abstractmethod
    def n_experiments(self):
        """
        The length of the experiment generator, aka the number of experiments.
        :return:
        """
        pass

    @staticmethod
    def tqdm_metrics_dict(result_dict: dict[str, Any]) -> dict[str, Any]:
        """
        This method returns a dictionary of any metrics to be logged at the tqdm progress bar.  Optionally override it
         for each experiment.
        """
        return {}
