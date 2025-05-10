import openai
import os
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import time
from tqdm import tqdm
from models import get_model 
import numpy as np 
import random
from datasets import get_dataset

def single_chat_llama(model_instance, tokenizer, questionnaire, questions_string, in_context_docs, in_context_examples_prompt_name):
    if in_context_docs != None:
        with open(os.path.join("prompts", "questionnaire", in_context_examples_prompt_name)) as p:
            prompt_template = p.read()
    questionnaire_part_of_prompt = questionnaire["prompt"] + '\n' + questions_string 

    # If control, just do the questionaire prompting, otherwise share the samples first
    prompt = (questionnaire_part_of_prompt if in_context_docs == None
              else prompt_template.format(in_context_docs=in_context_docs) + '\n\n' + questionnaire_part_of_prompt)
    
    # Applies the questionnaire-specific system prompt 
    chat = [
        {"role": "system", "content": questionnaire["inner_setting"]},
        {"role": "user", "content": prompt},
    ]
    
    context = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # tokenize the context
    input_tokens = tokenizer(context, return_tensors='pt').to(model_instance.device)
    n_input_tokens = input_tokens['input_ids'].size(-1)

    # generate output tokens
    #TODO make this dynamic, get Kyle's help for passing the model gen config stuff
    output_tokens = model_instance.generate(**input_tokens, max_new_tokens=2000, pad_token_id=tokenizer.eos_token_id)
    
    model_answer = tokenizer.decode(output_tokens[0][n_input_tokens:], skip_special_tokens=True)
    return context, model_answer


def subreddit_examples_generator(questionnaire, args):
    testing_file = args.testing_file
    model = args.model
    records_file = args.name_exp if args.name_exp is not None else os.path.basename(args.model)

    openai.api_key = args.openai_key

    # Read the existing CSV file into a pandas DataFrame
    df = pd.read_csv(testing_file)

    # Find the columns whose headers start with "order"
    order_columns = [col for col in df.columns if col.startswith("order")]
    shuffle_count = 0
    insert_count = 0
    total_iterations = len(order_columns) * args.test_count

    in_context_samples_prompt = args.in_context_samples_prompt
    subreddit_name = args.subreddit
    num_in_context_samples = args.num_in_context_samples
    num_comments = args.num_comments

    quantize = True if args.quantize != None else args.quantize
    use_flash_attn = True if args.use_flash_attn != None else args.use_flash_attn

    with tqdm(total=total_iterations) as pbar:
        for i, header in enumerate(df.columns):
            if header in order_columns:
                # Find the index of the previous column
                questions_column_index = i - 1
                shuffle_count += 1
                
                # Retrieve the column data as a string
                questions_list = df.iloc[:, questions_column_index].astype(str)
                separated_questions = [questions_list[i:i+30] for i in range(0, len(questions_list), 30)]  
                questions_list = ['\n'.join([f"{i+1}.{q.split('.')[1]}" for i, q in enumerate(questions)]) for j, questions in enumerate(separated_questions)]

                for k in range(args.test_count):
                    
                    df = pd.read_csv(testing_file)
                    
                    # Insert the updated column into the DataFrame with a unique identifier in the header
                    column_header = f'shuffle{shuffle_count - 1}-test{k}'
                    
                    while(True):
                        result_string_list = []
                        previous_records = []
                        
                        # If not the control, we should have a subreddit name to sample from
                        in_context_dataset = None if subreddit_name == None else get_dataset("reddit", split=subreddit_name, num_comments=num_comments) 

                        # Randomly sample a fixed number of n in-context samples, if it is NOT the control, and we want the template for the in-context content consumed
                        # Is done once at the start before the model is prompted in chunks of questions, so the example given are static
                        if in_context_dataset != None:
                            in_context_posts_idx = np.random.choice(len(in_context_dataset),
                                                                    size=num_in_context_samples,
                                                                    replace=False).tolist()
                            in_context_docs = "\n\n".join(
                                [f"PREVIOUS content chunk consumed:\n{in_context_dataset[position]['content']}" for
                                    i, position in enumerate(in_context_posts_idx)])
                        else:
                            in_context_docs = None

                        for questions_string in questions_list:
                            result = ''
                            if "llama" in model:
                                config = {'name': model, 'use_flash_attn': use_flash_attn, 'quantize': quantize}
                                model_instance, tokenizer = get_model(model_type="hf", **config)
                                
                                # Answers a chunk of questions at a time (questions_string), from my understanding
                                inputs, result = single_chat_llama(model_instance=model_instance, tokenizer=tokenizer, 
                                                                   questionnaire=questionnaire, questions_string=questions_string, 
                                                                   in_context_docs=in_context_docs,
                                                                   in_context_examples_prompt_name=in_context_samples_prompt)

                            elif model.startswith("gpt"):
                                inputs = previous_records + [
                                    {"role": "system", "content": questionnaire["inner_setting"]},
                                    {"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string}
                                ]
                                result = chat(model, inputs)
                                previous_records.append({"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string})
                                previous_records.append({"role": "assistant", "content": result})
                            else:
                                raise ValueError("The model is not supported or does not exist.")
                        
                            result_string_list.append(result.strip())
                        
                            # Write the prompts and results to the file
                            os.makedirs("psychobench/prompts", exist_ok=True)
                            os.makedirs("psychobench/responses", exist_ok=True)

                            with open(f'psychobench/prompts/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                                file.write(f'{inputs}\n====\n')
                            with open(f'psychobench/responses/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                                file.write(f'{result}\n====\n')

                        result_string = '\n'.join(result_string_list)
                        
                        result_list = convert_results(result_string, column_header)
                        
                        try:
                            if column_header in df.columns:
                                df[column_header] = result_list
                            else:
                                df.insert(i + insert_count + 1, column_header, result_list)
                                insert_count += 1
                            break
                        except:
                            print(f"Unable to capture the responses on {column_header}.")

                    # Write the updated DataFrame back to the CSV file
                    df.to_csv(testing_file, index=False)
                    
                    pbar.update(1)



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat(
    model,                      # gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301
    messages,                   # [{"role": "system"/"user"/"assistant", "content": "Hello!", "name": "example"}]
    temperature=0,    # [0, 2]: Lower values -> more focused and deterministic; Higher values -> more random.
    n=1,                        # Chat completion choices to generate for each input message.
    max_tokens=1024,            # The maximum number of tokens to generate in the chat completion.
    delay=1           # Seconds to sleep after each request.
):
    time.sleep(delay)
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        n=n,
        max_tokens=max_tokens
    )
    if n == 1:
        return response.choices[0].message.content
    else:
        return [i.message.content for i in response.choices]

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion(
    model,           # text-davinci-003, text-davinci-002, text-curie-001, text-babbage-001, text-ada-001
    prompt,          # The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.
    temperature=0,   # [0, 2]: Lower values -> more focused and deterministic; Higher values -> more random.
    n=1,             # Completions to generate for each prompt.
    max_tokens=1024, # The maximum number of tokens to generate in the chat completion.
    delay=1         # Seconds to sleep after each request.
):
    time.sleep(delay)
    
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        n=n,
        max_tokens=max_tokens
    )
    
    if n == 1:
        return response['choices'][0]['text']
    else:
        response = response['choices']
        response.sort(key=lambda x: x['index'])
        return [i['text'] for i in response['choices']]


def convert_results(result, column_header):
    result = result.strip()  # Remove leading and trailing whitespace
    try:
        result_list = [int(element.strip()[-1]) for element in result.split('\n') if element.strip()]
    except:
        result_list = ["" for element in result.split('\n')]
        print(f"Unable to capture the responses on {column_header}.")
        
    return result_list


def example_generator(questionnaire, args):
    testing_file = args.testing_file
    model = args.model
    records_file = args.name_exp if args.name_exp is not None else model

    openai.api_key = args.openai_key

    # Read the existing CSV file into a pandas DataFrame
    df = pd.read_csv(testing_file)

    # Find the columns whose headers start with "order"
    order_columns = [col for col in df.columns if col.startswith("order")]
    shuffle_count = 0
    insert_count = 0
    total_iterations = len(order_columns) * args.test_count

    with tqdm(total=total_iterations) as pbar:
        for i, header in enumerate(df.columns):
            if header in order_columns:
                # Find the index of the previous column
                questions_column_index = i - 1
                shuffle_count += 1
                
                # Retrieve the column data as a string
                questions_list = df.iloc[:, questions_column_index].astype(str)
                separated_questions = [questions_list[i:i+30] for i in range(0, len(questions_list), 30)]  
                questions_list = ['\n'.join([f"{i+1}.{q.split('.')[1]}" for i, q in enumerate(questions)]) for j, questions in enumerate(separated_questions)]


                for k in range(args.test_count):
                    
                    df = pd.read_csv(testing_file)
                    
                    # Insert the updated column into the DataFrame with a unique identifier in the header
                    column_header = f'shuffle{shuffle_count - 1}-test{k}'
                    
                    while(True):
                        result_string_list = []
                        previous_records = []
                        
                        for questions_string in questions_list:
                            result = ''
                            if model == 'text-davinci-003':
                                inputs = questionnaire["inner_setting"].replace('Format: \"index: score\"', 'Format: \"index: score\\\n\"') + questionnaire["prompt"] + '\n' + questions_string
                                result = completion(model, inputs)
                            elif model.startswith("gpt"):
                                inputs = previous_records + [
                                    {"role": "system", "content": questionnaire["inner_setting"]},
                                    {"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string}
                                ]
                                result = chat(model, inputs)
                                previous_records.append({"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string})
                                previous_records.append({"role": "assistant", "content": result})
                            else:
                                raise ValueError("The model is not supported or does not exist.")
                        
                            result_string_list.append(result.strip())
                        
                            # Write the prompts and results to the file
                            os.makedirs("psychobench/prompts", exist_ok=True)
                            os.makedirs("psychobench/responses", exist_ok=True)

                            with open(f'psychobench/prompts/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                                file.write(f'{inputs}\n====\n')
                            with open(f'psychobench/responses/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                                file.write(f'{result}\n====\n')

                        result_string = '\n'.join(result_string_list)
                        
                        result_list = convert_results(result_string, column_header)
                        print(result_list)
                        try:
                            if column_header in df.columns:
                                df[column_header] = result_list
                            else:
                                df.insert(i + insert_count + 1, column_header, result_list)
                                insert_count += 1
                            break
                        except:
                            print(f"Unable to capture the responses on {column_header}.")

                    # Write the updated DataFrame back to the CSV file
                    df.to_csv(testing_file, index=False)
                    
                    pbar.update(1)

