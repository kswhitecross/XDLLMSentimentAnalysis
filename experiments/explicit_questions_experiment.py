"""
This file contains an implementation of the explicit questions experiment.
"""
from datasets import get_dataset
import os
from .experiment import Experiment
from typing import Generator, Any
import numpy as np
import re
import random 

class ExplicitQuestionsExperiment(Experiment):
    """
     An experiment where the LLM is *explicitly* asked about its feelings/score on an inquiry example,
     based on a specified number of in-context examples. 
    """

    def __init__(self, model, tokenizer, *args, 
                d1_name: str = "sample",
                d1_split: str = "first",
                d2_name: str = "sample",
                d2_split: str = "second",
                prompt_name: str = "SampleExperimentPrompt.txt",
                num_in_context_samples: int = 2,
                max_generate: int = 256,
                 **kwargs):
        super().__init__("explicit_questions_experiment")
        self.tokenizer = tokenizer
        self.max_generate = max_generate
        self.num_in_context_samples = num_in_context_samples

        # create datasets
        if d1_name != None:
            self.in_context_dataset = get_dataset(d1_name, split=d1_split)
        else: 
             self.in_context_dataset = None

        self.inquiry_dataset = get_dataset(d2_name, split=d2_split)

        # load the prompt, which uses the control prompt on the second dataset if our in-context dataset is not provided
        with open(os.path.join('prompts', 
                               'explicit', 
                               'control' if self.in_context_dataset == None else 'experimental', 
                               prompt_name)) as p:
                    self.prompt_template = p.read()

        # load the explicit questions to randomly place in the prompt, with their indices tracked
        with open(os.path.join('data', 'questions', 'explicit_questions.txt')) as q:
                    raw_questions = q.readlines()
                    self.explicit_questions_with_idx = [(i, question.strip()) for i, question in enumerate(raw_questions)]

    def _get_experiment_generator(self) -> Generator[dict[str, Any], None, None]:
        """
        Create the experiment generator.
        """
        def gen():
            # for each document in the dataset of inquiry...
            for i, item in enumerate(self.inquiry_dataset):
                # if not the control, randomly sample dynamic number of documents from the in-context dataset and structure them for the prompt
                if self.in_context_dataset != None:
                    samp_idx = np.random.choice(len(self.in_context_dataset), size=self.num_in_context_samples, replace=False).tolist()
                    in_context_docs = "\n\n".join([f"IN-CONTEXT Document {i+1}:\n{self.in_context_dataset[position]['content']}" for i, position in enumerate(samp_idx)])

                # randomly order the explicit questions to add into the prompt template, without impacting the original order 
                shuffled_explicit_questions_with_idx = random.sample(self.explicit_questions_with_idx, len(self.explicit_questions_with_idx))
                question_dict = {f"question{i+1}": shuffled_explicit_questions_with_idx[i][1] for i in range(len(shuffled_explicit_questions_with_idx))}

                # create the prompt, depending on whether it's a control or not
                prompt = (
                     self.prompt_template.format(in_context_docs=in_context_docs, inq_doc=item['content'], **question_dict) if self.in_context_dataset != None 
                    else self.prompt_template.format(inq_doc=item['content'], **question_dict))
                
                # chat templatize the prompt, IF USIING the instruct model that has one
                # https://huggingface.co/docs/transformers/main/en/chat_templating
                
                # TODO: Determine the impact of the system prompt on our analysis/goals
                chat = [
                    {"role": "system", "content": "You are a helpful chatbot"},
                    {"role": "user", "content": prompt},
                ]
                context = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

                # build the test dict, with the variable number of in-context samples
                test_dict = {
                    "context": context,
                    "max_gen_tokens": self.max_generate,
                    "in_context_doc_indices": None if self.in_context_dataset == None else samp_idx,
                    "inq_doc_idx": i,
                    "prompt": prompt,
                    "question_order": [shuffled_explicit_questions_with_idx[i][0] for i in range(len(shuffled_explicit_questions_with_idx))]
                }
                
                yield test_dict
        return gen()

    def evaluate_results(self, result_dict: dict[str, Any]):
        """
        Process the model's answer, and add the results to the result_dict
        """
        ans = result_dict['model_answer']
        sentiment_distro = self.get_sentiment_from_pretrained_model(ans, self.sentiment_model_name)
        result_dict['sentiment_scores'] = sentiment_distro.tolist() 
        return


    @property
    def n_experiments(self):
        return len(self.inquiry_dataset)

    @staticmethod
    def tqdm_metrics_dict(result_dict: dict[str, Any]) -> dict[str, Any]:
        ret_dict = {
            "sentiment_scores": result_dict['sentiment_scores'],
        }
        return ret_dict