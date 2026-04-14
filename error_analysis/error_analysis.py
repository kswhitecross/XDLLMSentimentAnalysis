import json
import random 

model_names = ["llama_1B", "llama_3B", "llama_8B", "llama_70B"]
model_configurations = ["short", "long"]

for model_config in model_configurations:
    for model_name in model_names:
        results_json = []
        with open(f"initial_responses/{model_config}/{model_name}/results.jsonl", 'r') as file:
            for line in file:
                results_json.append(json.loads(line))
        #get the indices that we're going to save
        indices = random.sample(range(0, len(results_json)), 5)
        selected_results = {}
        for index in indices:
            selected_results[str(index)] = results_json[index]
        with open(f"selected_responses/{model_config}_{model_name}.json", 'w') as file:
            json.dump(selected_results, file, indent=4)