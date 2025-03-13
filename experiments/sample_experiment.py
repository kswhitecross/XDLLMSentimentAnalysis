"""
This file contains an implementation of a sample experiment.
"""
from datasets import get_dataset
import os
from .experiment import Experiment
from typing import Generator, Any
import numpy as np
import re


class SampleExperiment(Experiment):
    """
    A sample experiment, used to validate the correctness of existing code.  Prompts the LLM to score a document
     from dataset2 on a scale of 1-5 based on two examlpes from dataset1, with a justification.
    """

    def __init__(self, model, tokenizer, *args,
                 dataset1_name: str = "sample",
                 dataset1_split: str = "first",
                 dataset2_name: str = "sample",
                 dataset2_split: str = "second",
                 max_generate: int = 256,
                 **kwargs):
        super().__init__("SampleExperiment")
        self.tokenizer = tokenizer
        self.max_generate = max_generate

        # create datasets
        self.dataset1 = get_dataset(dataset1_name, split=dataset1_split)
        self.dataset2 = get_dataset(dataset2_name, split=dataset2_split)

        # load the prompt
        with open(os.path.join('prompts', 'SampleExperimentPrompt.txt')) as p:
            self.prompt_template = p.read()

    def _get_experiment_generator(self) -> Generator[dict[str, Any], None, None]:
        """
        Create the experiment generator.
        """
        def gen():
            # for each document in the second dataset...
            for i, item in enumerate(self.dataset2):
                # randomly sample 2 documents from the first dataset...
                samp_idx = np.random.choice(len(self.dataset2), size=2, replace=False).tolist()
                doc1 = self.dataset1[samp_idx[0]]
                doc2 = self.dataset2[samp_idx[1]]

                # create the prompt
                prompt = self.prompt_template.format(doc1=doc1['content'], doc2=doc2['content'],
                                                     inq_doc=item['content'])
                # chat templatize the prompt
                # https://huggingface.co/docs/transformers/main/en/chat_templating
                chat = [
                    {"role": "system", "content": "You are a helpful chatbot"},
                    {"role": "user", "content": prompt},
                ]
                context = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

                # build the test dict
                test_dict = {
                    "context": context,
                    "max_gen_tokens": self.max_generate,
                    "doc1_idx": samp_idx[0],
                    "doc2_idx": samp_idx[1],
                    "inq_doc_idx": i,
                    "prompt": prompt
                }
                yield test_dict
        return gen()

    def evaluate_results(self, result_dict: dict[str, Any]):
        """
        Process the model's answer, and add the results to the result_dict
        """
        ans = result_dict['model_answer']
        result_dict['score'] = None
        result_dict['justification'] = None

        # get the score
        score_rule = "\[\[Score\]\]: \d"
        if match := re.search(score_rule, ans):
            score = int(match.group().split(' ')[1])
            result_dict['score'] = score

        # get the justification
        # thanks chatgpt for this regex...
        just_rule = "\[\[Justification\]\]: ([\s\S]*?)\s*\[\[Score\]\]"
        if match := re.search(just_rule, ans):
            justification = match.group(1)
            result_dict['justification'] = justification
        return

    @property
    def n_experiments(self):
        return len(self.dataset2)
