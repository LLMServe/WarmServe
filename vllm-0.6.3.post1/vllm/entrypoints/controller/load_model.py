import os
import time
import ctypes
import argparse
from vllm.executor.ray_utils import ray
from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
from multiprocessing import shared_memory
from multiprocessing.shared_memory import SharedMemory

from .ModelConfig import ModelList
from .memory_utils import get_shm_name, wrap_ptr_to_tensor

runtime_start_time = time.time()

class TensorManager:
    def __init__(self, model_name, tp_rank):
        self.model_name = model_name
        self.tp_rank = tp_rank

        file_name = os.path.join(
            model_name,
            f"model-rank-{tp_rank}-part-0.safetensors"
        )

        weights_iterator = safetensors_weights_iterator([file_name])

        # Get total elements
        num_ele = 0
        weights = []
        for name, tensor in weights_iterator:
            num_ele += tensor.numel()
            weights.append((name, tensor))

        # Create shared memory
        buffer_size = num_ele * 2
        shm_name = get_shm_name(model_name, tp_rank)
        shm = SharedMemory(name=shm_name, create=True, size=buffer_size)        
        buffer_ptr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
        offset = 0

        self.state_dict = {}
        for name, tensor in weights:
            size = tensor.numel() * 2
            pin_tensor = wrap_ptr_to_tensor(buffer_ptr + offset, size).view(tensor.shape)
            pin_tensor.copy_(tensor)
            self.state_dict[name] = (offset, tensor.shape)
            offset += size
        
        self.buffer_size = buffer_size

@ray.remote(num_cpus=1)
class MemoryModelLoader:
    def __init__(self, only_create_mapping: bool = True, index: int = 0):
        os.system("rm /dev/shm/model-*")
        models = set()
        for task_type, model_dict in ModelList.items():
            for model_id, model_info in model_dict.items():
                if model_id not in models:
                    models.add((model_id, model_info[1]))

        self.only_create_mapping = only_create_mapping
        if only_create_mapping:
            self.shm_list = []
            counter = 0
            models = sorted(list(models))
            for model_name, tp_size in models:
                if index == 0:
                    # Create another model version whose tensors are in memory
                    new_dir = model_name + "_mem"
                    os.makedirs(new_dir, exist_ok=True)
                    for file in os.listdir(model_name):
                        if os.path.isfile(f"{model_name}/{file}"):
                            if not file.endswith(".safetensors") and not file.endswith(".bin"):
                                os.system(f"cp {model_name}/{file} {new_dir}/{file}")
                # Prepare model
                for tp_rank in range(tp_size):
                    file_name = os.path.join(
                        model_name,
                        f"model-rank-{tp_rank}-part-0.safetensors"
                    )
                    shm_name = f"model-{counter}"
                    counter += 1

                    with open(file_name, "rb") as f:
                        file_data = f.read()
                    
                    buffer_size = len(file_data)
                    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=buffer_size)

                    shm.buf[:buffer_size] = file_data

                    self.shm_list.append(shm)

                    if index == 0:
                        shm_file_path = f"/dev/shm/{shm_name}"
                        symlink_path = os.path.join(new_dir, f"model-rank-{tp_rank}-part-0.safetensors")
                        if os.path.exists(symlink_path) or os.path.islink(symlink_path):
                            os.unlink(symlink_path)  # Delete existing soft link
                        os.symlink(shm_file_path, symlink_path)
                print(f"Loaded model {model_name} with tp = {tp_size}")
        else:
            # Prepare model
            self.loaded_models = {}
            for model_name, tp_size in models:
                for tp_rank in range(tp_size):
                    self.loaded_models[(model_name, tp_rank)] = TensorManager(model_name, tp_rank)
                print(f"Loaded model {model_name} with tp = {tp_size}")

        print(f"Load models into shared memory complete. Time cost = {time.time() - runtime_start_time} seconds.")
    
    def get_dict(self, model_name: str, rank: int):
        if self.only_create_mapping:
            raise ValueError("The model loader only creates weight mapping!")
        tensor_manager = self.loaded_models[(model_name, rank)]
        return tensor_manager.buffer_size, tensor_manager.state_dict

    def wait_compl(self):
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-create-mapping', action='store_true', help='Only create mappings of model weights.')

    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)
    nodes = [node for node in ray.nodes() if node["Alive"]]
    node_ips = [node["NodeManagerAddress"] for node in nodes]
    workers = []
    for index, ip in enumerate(node_ips):
        worker = MemoryModelLoader.options(resources={f"node:{ip}": 0.01}, name=f"memory_model_manager_{ip}", namespace="prewarm", lifetime="detached").remote(args.only_create_mapping, index)
        workers.append(worker)
    
    for worker in workers:
        ray.get(worker.wait_compl.remote())

    print("Load models into memory complete.")