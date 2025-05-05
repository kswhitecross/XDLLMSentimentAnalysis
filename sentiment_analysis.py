import argparse
from models import get_hf_model
from abc import ABC, abstractmethod
import os
import json
import re
from tqdm import tqdm
from transformers import pipeline


class Model(ABC):
    @abstractmethod
    def apply_model(self, **prompt_kwargs) -> str:
        pass


class HFModel(Model):
    def __init__(self, model, tokenizer, prompt_template, system_prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt_template
        self.system_prompt = system_prompt

    def apply_model(self, **prompt_kwargs) -> str:
        chat = [
            {'role': 'system', 'content': self.system_prompt},
            {"role": "user", "content": self.prompt.format(**prompt_kwargs)}
        ]
        input_tokens = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors='pt',
                                                          tokenize=True, return_dict=True).to(self.model.device)
        n_input_tokens = len(input_tokens['input_ids'])
        output_tokens = self.model.generate(**input_tokens, pad_token_id=self.tokenizer.eos_token_id)[0]
        output = self.tokenizer.decode(output_tokens[n_input_tokens:], skip_special_tokens=True)
        return output


class GPTModel(Model):
    def __init__(self, prompt_template: str, system_prompt: str):
        self.prompt = prompt_template
        self.system_prompt = system_prompt

    def apply_model(self, **prompt_kwargs) -> str:
        return "hello!"


def count_lines(filename):
    """
    Counts the number of lines in filename
    """
    count = 0
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            count += chunk.count(b'\n')
    return count


def parse_sentiment_response(model_response: str) -> dict:
    """
    Parses the sentiment analysis output from the model.  Returns a dict with 'justification' and
     'score' as keys.
    """
    just_search = re.search("Justification: ([\s\S]*?)\s*Score:", model_response)
    score_search = re.search("Score: (\d+)", model_response)
    ret_dict = {
        'justification': None,
        'score': None
    }
    if just_search:
        ret_dict['justification'] = just_search.group(1)
    if score_search:
        ret_dict['score'] = score_search.group(1)
    return ret_dict


def main(args: argparse.Namespace):
    # read in prompts
    with open(args.system_prompt, 'r') as p:
        system_prompt = p.read()
    with open(args.prompt, 'r') as p:
        prompt_template = p.read()

    # load the model
    print("Loading model...")
    if args.use_gpt:
        chatbot = GPTModel(prompt_template, system_prompt)
    else:
        # get llama 3.3 quantized otherwise
        gen = {
            "do_sample": True,
            "max_new_tokens": 2000,
            "temperature": 0.6,
            "top_p": 0.9,
        }
        model, tokenizer = get_hf_model(
            name='meta-llama/Llama-3.1-8B-Instruct',
            use_flash_attn=True,
            quantize=False,
            gen=gen
        )
        # retroactively set the pad_token_id to the eos_token_id to suppress warnings
        chatbot = HFModel(model, tokenizer, prompt_template, system_prompt)


    # find all folders in runs_dir with a results.jsonl in them
    runs = [os.path.join(args.runs_dir, run) for run in os.listdir(args.runs_dir)
            if os.path.exists(os.path.join(args.runs_dir, run, 'results.jsonl'))]

    # iterate through every run
    for run in runs:
        # compute the number of lines
        results_filename = os.path.join(run, 'results.jsonl')
        n_lines = count_lines(results_filename)
        print(run)
        with open(results_filename, 'r') as results_file:
            with open(os.path.join(run, 'sentiment.jsonl'), 'w') as sentiment_file:
                for line in tqdm(results_file, total=n_lines):
                    # read in the line
                    line_dict = json.loads(line)
                    model_question_answer = line_dict['model_answer']

                    # pass the answer through the model
                    sentiment_model_response = chatbot.apply_model(model_answer=model_question_answer)

                    # parse the response
                    sent_dict = parse_sentiment_response(sentiment_model_response)

                    # save outputs
                    sentiment_file.write(json.dumps(sent_dict) + "\n")
                    sentiment_file.flush()

    # done!
    print("Done!")


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(description="Run LLM zero-shot sentiment analysis on existing responses. Saves the"
                                                 "results as `sentiment.jsonl` in the same folder as `results.jsonl`")
    parser.add_argument("--runs_dir", type=str, default='runs/reddit/implicit/original',
                        help='Location of subfolders containing `results.jsonl` files.')
    parser.add_argument("--use_gpt", type=bool, default=False, help="Use ChatGPT 4 to generate the "
                        "responses")
    parser.add_argument("--prompt", type=str, default='prompts/SentimentAnalysis.txt',
                        help="Prompt filepath")
    parser.add_argument("--system_prompt", type=str, default='prompts/SentimentAnalysisSystem.txt',
                        help="System prompt filepath")
    parsed_args = parser.parse_args()
    main(parsed_args)
