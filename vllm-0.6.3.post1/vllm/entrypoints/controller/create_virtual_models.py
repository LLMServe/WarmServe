import os

from .ModelConfig import ModelList

def create_sym_link(origin_model_path: str, dest_model_path: str):
    if os.path.exists(dest_model_path):
        print(f"Path {dest_model_path} already exists. Skip.")
    else:
        os.system(f"ln -s {origin_model_path} {dest_model_path}")
        print(f"Created symbolic link from {origin_model_path} to {dest_model_path}")

if __name__ == "__main__":
    model_cur_index = {}
    for task_type, model_dict in ModelList.items():
        for model_id, model_info in model_dict.items():
            os.makedirs(model_id + "_mem", exist_ok=True)
            if model_id not in model_cur_index:
                model_cur_index[model_id] = 0
                cur_index = 0
            else:
                cur_index = model_cur_index[model_id]
            for index in range(model_info[2]):
                split_model_id = model_id + "/" + str(index + cur_index)
                create_sym_link(model_id, split_model_id)
                split_model_id_mem = model_id + "_mem/" + str(index + cur_index)
                create_sym_link(model_id + "_mem", split_model_id_mem)
            model_cur_index[model_id] += model_info[2]