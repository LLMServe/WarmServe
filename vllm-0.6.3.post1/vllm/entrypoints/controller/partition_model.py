import argparse

import vllm
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader.loader import DefaultModelLoader
from safetensors.torch import save_file

import os
import torch

# Overwrite tp && pp functions
class MyGroupCoordinator:
    def __init__(self):
        self.rank = 0
        self.size = 1
    
    @property
    def is_first_rank(self):
        return self.rank == 0

    @property
    def is_last_rank(self):
        return self.rank == self.size - 1
    
    @property
    def rank_in_group(self):
        return self.rank
    
    @property
    def world_size(self):
        return self.size

ppGroup = MyGroupCoordinator()
tpGroup = MyGroupCoordinator()

from vllm.distributed import parallel_state
parallel_state._PP[""] = ppGroup
parallel_state._TP[""] = tpGroup

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--pipeline-parallel-size', type=int, default=1)

    args = parser.parse_args()
    tp_size = args.tensor_parallel_size
    pp_size = args.pipeline_parallel_size
    model_path_ = os.path.join(args.model_path, f"tp{tp_size}_pp{pp_size}")
    os.makedirs(model_path_, exist_ok=True)

    for file in os.listdir(args.model_path):
        if os.path.isfile(f"{args.model_path}/{file}"):
            if not file.endswith("safetensors") and not file.endswith("pt"):
                os.system(f"cp {args.model_path}/{file} {model_path_}/{file} ")

    try:
        engine_args = EngineArgs(model=args.model_path, tensor_parallel_size=tp_size, pipeline_parallel_size=pp_size, device="cpu", dtype="float16")
        engine_config = engine_args.create_engine_config()
        loader = DefaultModelLoader(engine_config.load_config)

        world_size = tp_size * pp_size
        for rank in range(world_size):
            pp_rank = rank // tp_size
            tp_rank = rank % tp_size

            tpGroup.rank = tp_rank
            tpGroup.size = tp_size
            ppGroup.rank = pp_rank
            ppGroup.size = pp_size

            model = loader.load_model(
                model_config=engine_config.model_config,
                device_config=engine_config.device_config,
                lora_config=engine_config.lora_config,
                parallel_config=engine_config.parallel_config,
                scheduler_config=engine_config.scheduler_config,
                cache_config=engine_config.cache_config,
            )

            file_name = os.path.join(model_path_, f"model-rank-{rank}-part-0.safetensors")
            save_file(model.state_dict(), file_name)
            print(f"Partition success for tp = {tp_rank}, pp = {pp_rank}, path = {file_name}.")
    except Exception as e:
        print(f"Partition model with tp = {tp_size}, pp = {pp_size} meet exception {e}.")