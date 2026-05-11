import os
import sys
import yaml

from .ModelConfig import ModelList

save_path = sys.argv[1]

num_nodes = int(os.getenv("NUM_NODES", "1"))
num_gpu_per_server = int(os.getenv("NUM_GPUS_PER_NODE", "8"))

output_dict = {'cluster': {'nnodes': num_nodes, 'ngpus_per_node': num_gpu_per_server}}

models = []

model_cur_index = {}
model_id = 0
for task_type, model_dict in ModelList.items():
    for model_id, model_info_ in model_dict.items():
        if model_id not in model_cur_index:
            model_cur_index[model_id] = 0
            cur_index = 0
        else:
            cur_index = model_cur_index[model_id]
        # Remove /tp1_pp1 suffix
        pos = model_id.rfind('/')
        orig_model_path = model_id[:pos]
        for index in range(model_info_[2]):
            split_model_id = model_id + "/" + str(index + cur_index)
            datasource = model_info_[3]
            models.append({"name": split_model_id, "id": f"llm-{model_id}", "dataset_source": datasource})
            model_id += 1
        model_cur_index[model_id] += model_info_[2]

output_dict["models"] = models
with open(save_path, "w") as f:
    yaml.dump(output_dict, f)

print(f"Yaml generated to {save_path}.")