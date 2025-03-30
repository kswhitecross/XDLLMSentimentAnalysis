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

        # load the prompt
        with open(os.path.join('prompts', 
                               'explicit_questions_attempts', 
                               'control' 
                               if self.in_context_dataset == None else 'experimental', 
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
                # randomly sample 2 documents from the in-context dataset...
                samp_idx = np.random.choice(len(self.in_context_dataset), size=self.num_in_context_samples, replace=False).tolist()

                # TODO dynamically label a bunch of variables to put in the prompt template
                doc1 = self.in_context_dataset[samp_idx[0]]
                doc2 = self.in_context_dataset[samp_idx[1]]
                
                # randomly order the explicit questions to add into the prompt template, without impacting the original order 
                shuffled_explicit_questions_with_idx = random.sample(self.explicit_questions_with_idx, len(self.explicit_questions_with_idx))
                question_dict = {f"question{i+1}": shuffled_explicit_questions_with_idx[i][1] for i in range(len(shuffled_explicit_questions_with_idx))}

                # create the prompt
                prompt = self.prompt_template.format(doc1=doc1['content'], doc2=doc2['content'],
                                                     inq_doc=item['content'], **question_dict)
                
                # chat templatize the prompt, IF USIING the instruct model that has one
                # https://huggingface.co/docs/transformers/main/en/chat_templating
                chat = [
                    {"role": "system", "content": "You are a helpful chatbot"},
                    {"role": "user", "content": prompt},
                ]
                context = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

                # build the test dict, with the variable number of in-context samples
                test_dict = {
                    "context": context,
                    "max_gen_tokens": self.max_generate,
                    "doc1_idx": samp_idx[0],
                    "doc2_idx": samp_idx[1],
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
        result_dict['rating'] = None
        result_dict['justification'] = None

        # get the score
        score_rule = "\[\[Rating\]\]: \d"
        if match := re.search(score_rule, ans):
            score = int(match.group().split(' ')[1])
            result_dict['rating'] = score

        # get the justification
        # thanks chatgpt for this regex...
        just_rule = "\[\[Justification\]\]: ([\s\S]*?)\s*\[\[Score\]\]"
        if match := re.search(just_rule, ans):
            justification = match.group(1)
            result_dict['justification'] = justification
        return

    @property
    def n_experiments(self):
        return len(self.inquiry_dataset)

    @staticmethod
    def tqdm_metrics_dict(result_dict: dict[str, Any]) -> dict[str, Any]:
        ret_dict = {
            "rating": result_dict['rating'],
        }
        return ret_dict