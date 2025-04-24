"""
This file contains an implementation of a sample experiment.
"""

import subprocess
import re
from typing import Generator, Any
from datasets import books, news, youtube
import os
from .experiment import Experiment
import numpy as np


class BookSummaryWithContextExperiment:
    def __init__(self, model, tokenizer, inquiry_dataset, in_context_dataset, max_generate: int = 1024, max_inquiries: int = 1):
        self.tokenizer = tokenizer
        self.max_generate = max_generate
        self.inquiry_dataset = inquiry_dataset  
        self.context_dataset = in_context_dataset  
        self.max_inquiries = max_inquiries
        
        # load prompt template
        with open('prompts/experimental.txt', 'r') as f:
            self.prompt_template = f.read()


    def generate_response(self, prompt: str) -> str:
        print(f"Running command: ['ollama', 'run', 'llama-3', prompt]")
        result = subprocess.run(
            ["ollama", "run", "llama-3", prompt], capture_output=True, text=True
        )
        print(f"Model Output: {result.stdout}")
        return result.stdout

    def _get_experiment_generator(self) -> Generator[dict[str, Any], None, None]:
        def gen():
            # use only first book summary entry 
            inquiry_item = self.inquiry_dataset[0]

            # book summary inquiry doc
            inq_doc = inquiry_item['summary']

                
            in_context_docs = []
            # using 10 in-context youtube docs
            for _ in range(10):
                context_idx = np.random.choice(len(self.context_dataset), size=1).tolist()
                context_doc = self.context_dataset[context_idx[0]]["comment"]
                in_context_docs.append(context_doc)
                
            in_context_docs_str = "\n".join(in_context_docs)

            # questions to ask model
            question1 = "What is the main theme of the book summary?"
            question2 = "Based on the IN-CONTEXT documents, would the user be interested in reading the INQUIRY document?"
            question3 = "What emotions does the book summary evoke?"

            # create prompt
            prompt = self.prompt_template.format(
                in_context_docs=in_context_docs_str, 
                inq_doc=inq_doc, 
                question1=question1, 
                question2=question2, 
                question3=question3
            )

            # get model response
            model_answer = self.generate_response(prompt)
            # debug
            print(f"Raw Model Output: {model_answer}") 

            result_dict = {'model_answer': model_answer, 'prompt': prompt}
            self.evaluate_results(result_dict)
            yield result_dict

            # for i, inquiry_item in enumerate(self.inquiry_dataset):
            #     # book summary inquiry doc
            #     inq_doc = inquiry_item['summary']

                
            #     in_context_docs = []
            #     # using 3 in-context youtube docs
            #     for _ in range(3):
            #         context_idx = np.random.choice(len(self.context_dataset), size=1).tolist()
            #         context_doc = self.context_dataset[context_idx[0]]["comment"]
            #         in_context_docs.append(context_doc)
                
            #     in_context_docs_str = "\n".join(in_context_docs)

            #     # questions to ask model
            #     question1 = "What is the main theme of the book summary?"
            #     question2 = "How would you describe the writing style of the author?"
            #     question3 = "What emotions does the book summary evoke?"

            #     # create prompt
            #     prompt = self.prompt_template.format(
            #         in_context_docs=in_context_docs_str, 
            #         inq_doc=inq_doc, 
            #         question1=question1, 
            #         question2=question2, 
            #         question3=question3
            #     )

            #     # get model response
            #     model_answer = self.generate_response(prompt)

            #     result_dict = {'model_answer': model_answer, 'prompt': prompt}
            #     self.evaluate_results(result_dict)
            #     yield result_dict

        return gen()

    def evaluate_results(self, result_dict: dict[str, Any]):
        ans = result_dict['model_answer']
        result_dict['justification'] = None
        result_dict['rating'] = None

        # get model justification and rating
        just_rule = "\[\[Justification\]\]: ([\s\S]*?)\s*\[\[Rating\]\]"
        if match := re.search(just_rule, ans):
            justification = match.group(1)
            result_dict['justification'] = justification
            
            rating_rule = "\[\[Rating\]\]: (\d)"
            if match := re.search(rating_rule, ans):
                rating = int(match.group(1))
                result_dict['rating'] = rating

    @property
    def n_experiments(self):
        return len(self.inquiry_dataset)

    @staticmethod
    def tqdm_metrics_dict(result_dict: dict[str, Any]) -> dict[str, Any]:
        return {"rating": result_dict['rating'], "justification": result_dict['justification']}




# # from datasets import get_dataset
# from ..datasets import books, news, youtube
# import os
# from .experiment import Experiment
# from typing import Generator, Any
# import numpy as np
# import re


# class SampleExperiment(Experiment):
#     """
#     A sample experiment, used to validate the correctness of existing code.  Prompts the LLM to score a document
#      from dataset2 on a scale of 1-5 based on two examples from dataset1, with a justification.
#     """

#     def __init__(self, model, tokenizer, *args,
#                  d1_name: str = "sample",
#                  d1_split: str = "first",
#                  d2_name: str = "sample",
#                  d2_split: str = "second",
#                  max_generate: int = 256,
#                  **kwargs):
#         super().__init__("SampleExperiment")
#         self.tokenizer = tokenizer
#         self.max_generate = max_generate

#         # create datasets
#         self.dataset1 = get_dataset(d1_name, split=d1_split)
#         self.dataset2 = get_dataset(d2_name, split=d2_split)

#         # load the prompt
#         with open(os.path.join('prompts', 'SampleExperimentPrompt.txt')) as p:
#             self.prompt_template = p.read()

#     def _get_experiment_generator(self) -> Generator[dict[str, Any], None, None]:
#         """
#         Create the experiment generator.
#         """
#         def gen():
#             # for each document in the second dataset...
#             for i, item in enumerate(self.dataset2):
#                 # randomly sample 2 documents from the first dataset...
#                 samp_idx = np.random.choice(len(self.dataset2), size=2, replace=False).tolist()
#                 doc1 = self.dataset1[samp_idx[0]]
#                 doc2 = self.dataset2[samp_idx[1]]

#                 # create the prompt
#                 prompt = self.prompt_template.format(doc1=doc1['content'], doc2=doc2['content'],
#                                                      inq_doc=item['content'])
#                 # chat templatize the prompt
#                 # https://huggingface.co/docs/transformers/main/en/chat_templating
#                 chat = [
#                     {"role": "system", "content": "You are a helpful chatbot"},
#                     {"role": "user", "content": prompt},
#                 ]
#                 context = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

#                 # build the test dict
#                 test_dict = {
#                     "context": context,
#                     "max_gen_tokens": self.max_generate,
#                     "doc1_idx": samp_idx[0],
#                     "doc2_idx": samp_idx[1],
#                     "inq_doc_idx": i,
#                     "prompt": prompt
#                 }
#                 yield test_dict
#         return gen()

#     def evaluate_results(self, result_dict: dict[str, Any]):
#         """
#         Process the model's answer, and add the results to the result_dict
#         """
#         ans = result_dict['model_answer']
#         result_dict['score'] = None
#         result_dict['justification'] = None

#         # get the score
#         score_rule = "\[\[Score\]\]: \d"
#         if match := re.search(score_rule, ans):
#             score = int(match.group().split(' ')[1])
#             result_dict['score'] = score

#         # get the justification
#         # thanks chatgpt for this regex...
#         just_rule = "\[\[Justification\]\]: ([\s\S]*?)\s*\[\[Score\]\]"
#         if match := re.search(just_rule, ans):
#             justification = match.group(1)
#             result_dict['justification'] = justification
#         return

#     @property
#     def n_experiments(self):
#         return len(self.dataset2)

#     @staticmethod
#     def tqdm_metrics_dict(result_dict: dict[str, Any]) -> dict[str, Any]:
#         ret_dict = {
#             "score": result_dict['score'],
#         }
#         return ret_dict