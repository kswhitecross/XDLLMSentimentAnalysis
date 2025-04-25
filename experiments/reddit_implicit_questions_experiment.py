"""
This file contains an implementation of the implicit questions experiment.
"""
from datasets import get_dataset
from .experiment import Experiment

from typing import Generator, Any
import numpy as np
import os
import re
import random 

class RedditImplicitQuestionsExperiment(Experiment):
    """
     An experiment where the LLM is *implicitly* asked about its feelings/score on an inquiry example,
     based on a specified number of in-context examples. 
    """

    def __init__(self, 
                model,
                tokenizer,
                *args, 

                prompt_name: str = "SampleExperimentPrompt.txt", 

                num_in_context_samples: int = 2, 
                num_inquiry_samples: int = None,
                num_outputs_per_prompt: int = 1, 
                num_comments: int = 3,

                max_generate: int = 256, 
                **kwargs):
        
        super().__init__("implicit_questions_experiment")

        self.tokenizer = tokenizer
        self.max_generate = max_generate

        self.num_in_context_samples = num_in_context_samples
        self.num_inquiry_samples = num_inquiry_samples
        self.num_outputs_per_prompt = num_outputs_per_prompt 

        self.subreddits = ["funny", "askreddit", "gaming", "worldnews", "todayilearned", "awww", "music", "memes", "movies", "showerthoughts"]
        self.subreddits_data = [get_dataset("reddit", split=subreddit_name, num_comments=num_comments) for subreddit_name in self.subreddits]

        # Determine how many inquiry samples per subreddit
        if self.num_inquiry_samples is None:
            inquiry_sample_counts = [len(data) for data in self.subreddits_data]
        else:
            inquiry_sample_counts = [min(self.num_inquiry_samples, len(data)) for data in self.subreddits_data]

        # Total experiments = sum over inquiry subreddits of:
        # (inquiry_samples * 11 context options * outputs per prompt)
        self.num_experiments = sum(
            num_inquiry_posts * (len(self.subreddits) + 1) * self.num_outputs_per_prompt for num_inquiry_posts in inquiry_sample_counts
        )

        # Load both the experimental and control prompt templates
        with open(os.path.join('prompts', 'implicit', 'experimental', prompt_name)) as p:
                            self.experimental_prompt_template =  p.read()

        with open(os.path.join('prompts', 'implicit', 'control', prompt_name)) as p:  
                            self.control_prompt_template =  p.read()

        # Load the implicit question(s) to randomly place in the prompt, with their indices tracked
        with open(os.path.join('data', 'questions', 'implicit_questions.txt')) as q:
                    raw_questions = q.readlines()
                    self.implicit_questions_with_idx = [(i, question.strip()) for i, question in enumerate(raw_questions)]
        


    def _get_experiment_generator(self) -> Generator[dict[str, Any], None, None]:
        """
        Create the experiment generator.
        """
        def gen():
            # For this inquiry subreddit
            for inquiry_data_idx, inquiry_subreddit in enumerate(self.subreddits):

                # Get the posts to inquire about
                inquiry_data = self.subreddits_data[inquiry_data_idx]
                
                # For each post to inquire about
                num_inquiry_samples_used = 0
                for inquiry_post_idx, inquiry_post in enumerate(inquiry_data):
                    
                    # For each possible in-context domain to use for the inquiry post (and the control, represented as None), including its own domain 
                    for in_context_data_idx, in_context_subreddit in enumerate(self.subreddits + [None]):
                        
                        # Randomly sample n in-context samples, if it is NOT the control
                        if in_context_subreddit != None:
                            in_context_data = self.subreddits_data[in_context_data_idx]
                            in_context_posts_idx = np.random.choice(len(in_context_data), size=self.num_in_context_samples, replace=False).tolist()
                            in_context_docs = "\n\n".join([f"PREVIOUS content chunk consumed:\n{in_context_data[position]['content']}" for i, position in enumerate(in_context_posts_idx)])

                        # Randomly order the implicit questions to add into the prompt template, without impacting the original order 
                        shuffled_implicit_questions_with_idx = random.sample(self.implicit_questions_with_idx, len(self.implicit_questions_with_idx))
                        
                        # If only one question, don't add the numbering
                        if len(shuffled_implicit_questions_with_idx) < 2:
                            questions = f"Question:\n{shuffled_implicit_questions_with_idx[0][1]}"
                        else: 
                            questions = "\n\n".join([f"Question{i+1}:\n{shuffled_implicit_questions_with_idx[i][1]}" for i in range(len(shuffled_implicit_questions_with_idx))])
                        
                        # Create the prompt, depending on whether it's a control or not
                        prompt = (
                            self.experimental_prompt_template.format(in_context_docs=in_context_docs, inq_doc=inquiry_post['content'], questions=questions) if in_context_subreddit != None 
                            else self.control_prompt_template.format(inq_doc=inquiry_post['content'], questions=questions))
                        
                        # Chat templatize the prompt, IF USING the instruct model that has one
                        # https://huggingface.co/docs/transformers/main/en/chat_templating
                        chat = [
                            {"role": "system", "content": "You are a helpful chatbot"},
                            {"role": "user", "content": prompt},
                        ]

                        context = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

                        # Produce the test dict for as many times that we want to generate an output for this prompt
                        for o in range(self.num_outputs_per_prompt):
                            test_dict = {
                                "in_context_domain": in_context_subreddit,
                                "inquiry_domain": inquiry_subreddit, 

                                "in_context_doc_indices": None if in_context_subreddit == None else in_context_posts_idx,
                                "inq_doc_idx": inquiry_post_idx,

                                "question_order": [shuffled_implicit_questions_with_idx[i][0] for i in range(len(shuffled_implicit_questions_with_idx))],
                                
                                "context": context,
                                "prompt": prompt,
                                "max_gen_tokens": self.max_generate,

                                "output_instance_number": o + 1
                            }
                            yield test_dict

                # Stop when we have inquired about as many inquiry samples as specified if given
                num_inquiry_samples_used += 1  
                if self.num_inquiry_samples != None and self.num_inquiry_samples == num_inquiry_samples_used:
                        break 
        return gen()

    def evaluate_results(self, result_dict: dict[str, Any]):
        """
        Process the model's answer, and add the results to the result_dict
        """
        ans = result_dict['model_answer']
        # TODO extract its score from the JSON it produces, when scoring prompt is used
        return

    @property
    def n_experiments(self):
        return self.num_experiments

    @staticmethod
    def tqdm_metrics_dict(result_dict: dict[str, Any]) -> dict[str, Any]:
        ret_dict = {
            # "sentiment_scores": result_dict['sentiment_scores'],
        }
        return ret_dict