from argparse import ArgumentParser
from tqdm import tqdm
import os
import warnings
import time
import json

from config import CFG, update_config, finalize_config
from models import get_model
from experiments import get_experiment


def main():
    # create model, tokenizer
    print("Loading model...")
    model, tokenizer = get_model(CFG.model.type, CFG.model.name, CFG.model.use_flash_attn, CFG.model.quantize)

    # create experiment
    print("Creating Experiment")
    experiment = get_experiment(CFG.exp.name, model, tokenizer, **CFG.exp)

    # initialize experiment
    print("Starting Experiment")
    experiment_iter = experiment.create_experiment()

    # create results file
    results_path = os.path.join(CFG.save_path, 'results.jsonl') if not CFG.dont_save else '/dev/null'
    with open(results_path, 'w') as results_file:
        # create tqdm progress bar
        pbar = tqdm(experiment_iter, desc='Running experiment', total=experiment.n_experiments)
        num_generated_responses = 0 
        for test_dict in pbar:
            context = test_dict['context']
            max_generate = test_dict['max_gen_tokens']

            # tokenize the context
            input_tokens = tokenizer(context, return_tensors='pt').to(model.device)
            n_input_tokens = input_tokens['input_ids'].size(-1)
            start_time = time.time()

            # generate output tokens
            output_tokens = model.generate(**input_tokens, max_new_tokens=max_generate, pad_token_id=tokenizer.eos_token_id)
            m_time = time.time() - start_time
            test_dict['gen_time'] = m_time

            # evaluate output
            model_answer = tokenizer.decode(output_tokens[0][n_input_tokens:], skip_special_tokens=True)
            test_dict['model_answer'] = model_answer

            # debugging
            if CFG.verbose:
                print("=" * 20)
                print("Model input:")
                print(context)
                print("Model output:")
                print(model_answer)
                print("=" * 20)
            experiment.evaluate_results(test_dict)

            # update tqdm
            postfix = experiment.tqdm_metrics_dict(test_dict)
            pbar.set_postfix(postfix)

            # save results
            if not CFG.save_context:
                del test_dict['context']
                del test_dict['prompt']
            results_file.write(json.dumps(test_dict) + '\n')
            results_file.flush()
            num_generated_responses += 1

            if CFG.short_circuit or num_generated_responses == CFG.exp.max_num_generated_responses:
                break

    # create completed file
    with open(os.path.join(CFG.save_path, 'completed.txt'), 'w') as _:
        pass

    # done!
    print("Done!")


if __name__ == "__main__":
    # create argument parser
    parser = ArgumentParser("Cross Domain Sentiment Analysis experimental framework.")
    parser.add_argument("-c", "--config", default='configs/default.yaml',
                        help='The filepath of the .yaml config file')
    parser.add_argument("--dont_save", action='store_true', help='Don\'t save the results.')

    # parse args, and store other args in opts
    args, opts = parser.parse_known_args()

    # update the config and finalize it
    update_config(CFG, args.config, opts, args.dont_save)
    finalize_config(CFG)

    # if we want to save the results, create a new folder
    if not CFG.dont_save:
        if os.path.exists(CFG.save_path):
            warnings.warn(f"Run save path {CFG.save_path} already exists!")
        else:
            os.makedirs(CFG.save_path)

        # save the config
        with open(os.path.join(CFG.save_path, 'config.yaml'), 'w') as f:
            f.write(CFG.dump())

    # print config
    print(CFG.dump())

    # start the program
    main()
