from typing import Optional, List, Tuple, Dict

import torch
from torch import nn

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig)
from vllm.model_executor.model_loader.loader import (BaseModelLoader,
                                                     get_model_loader)
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name, get_model_architecture)


def get_model(*, model_config: ModelConfig, load_config: LoadConfig,
              device_config: DeviceConfig, parallel_config: ParallelConfig,
              scheduler_config: SchedulerConfig,
              lora_config: Optional[LoRAConfig],
              cache_config: CacheConfig,
              loaded_tensors: Optional[Dict[str, torch.Tensor]] = None) -> nn.Module:
    loader = get_model_loader(load_config)
    args = dict(
        model_config=model_config,
        device_config=device_config,
        lora_config=lora_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config)
    if loaded_tensors is not None:
        args.update(loaded_tensors=loaded_tensors)
    return loader.load_model(**args)


__all__ = [
    "get_model", "get_model_loader", "BaseModelLoader",
    "get_architecture_class_name", "get_model_architecture"
]
