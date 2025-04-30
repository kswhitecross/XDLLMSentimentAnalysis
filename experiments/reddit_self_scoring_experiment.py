"""
This file contains an implementation of the implicit questions experiment.
"""
from .experiment import Experiment
from typing import Generator, Any
import numpy as np
import os

class RedditSelfScoringExperiment(Experiment):
    """
     An experiment where the LLM is asked to continue the conversation from previous outputs and provide a sentiment score
    """

    def __init__(
            self,
            model,
            tokenizer,
            *args,
            prompt_name: str = "SampleExperimentPrompt.txt",
            num_outputs_per_prompt: int = 1,
            **kwargs):
        
        super().__init__("reddit_self_scoring_experiment")
        self.tokenizer = tokenizer
        self.num_outputs_per_prompt = num_outputs_per_prompt

        self.folder_of_contexts_to_append = 'llama_1b/llama_1B_4fc69d73e8c340f79e8fdd8e0abdc2c3/outputs'

        self.num_experiments = 0
        for _ in os.listdir(self.folder_of_contexts_to_append):
            self.num_experiments += 1

        
        # Load the prompt requesting a JSON formatted score
        with open(os.path.join('prompts', 'implicit', 'experimental', prompt_name)) as p:
            self.score_prompt_to_append = p.read()

    def _get_experiment_generator(self) -> Generator[dict[str, Any], None, None]:
        """
        Create the experiment generator.
        """
        def gen():
            # Iterate over all of the already generated outputs
            total = 0
            for context_file_name in os.listdir(self.folder_of_contexts_to_append):
                # For every context saved to text files, append new sentiment score request
                if context_file_name.endswith('.txt'):
                    context_file_path = os.path.join(self.folder_of_contexts_to_append, context_file_name)
                    
                    with open(context_file_path, 'r') as file:
                        context_output = file.read() 
                        # chat = [
                        #     {"role": "user", "content": self.score_prompt_to_append},
                        # ]

                        # context = context_output + self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

                        context = f'{context_output}\n{self.score_prompt_to_append}'
                        
                    
                        # Produce the test dict for as many times that we want to generate an output for this prompt
                        for o in range(self.num_outputs_per_prompt):
                            test_dict = {
                                "context": context,
                                "prompt": self.score_prompt_to_append,
                                "output_instance_number": o + 1
                            }
                            yield test_dict
                    total += 1
                if total == 10:
                    break

        return gen()

    @property
    def n_experiments(self):
        return self.num_experiments

    @staticmethod
    def tqdm_metrics_dict(result_dict: dict[str, Any]) -> dict[str, Any]:
        ret_dict = {
           
        }
        return ret_dict
