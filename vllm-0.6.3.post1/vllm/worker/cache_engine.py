"""CacheEngine class for managing the KV cache."""
from typing import List, Optional, Tuple

import torch

from vllm import envs
from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size,
                        is_pin_memory_available)

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
        rank: int = 0,
        kv_caches: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(self.head_size,
                                             model_config.get_sliding_window(),
                                             model_config.dtype,
                                             cache_config.cache_dtype,
                                             self.block_size,
                                             model_config.is_attention_free)
        
        self.enable_kv_prewarming = envs.VLLM_KV_PREWARMING
        if self.enable_kv_prewarming:
            assert kv_caches, "KV Prewarming needs pre-allocated kv cache space"
        
        # Initialize the cache.
        if kv_caches:
            gpu_cache, cpu_cache = kv_caches
        else:
            gpu_cache = None
            cpu_cache = None
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type, rank, gpu_cache)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu", rank, cpu_cache)

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
        rank: int = 0,
        input_kv_caches: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []

        if input_kv_caches is not None:
            if self.dtype != torch.float16 and self.dtype != torch.bfloat16:
                raise ValueError(f"KV cache prewarming only supports float16 and bfloat16, get {self.dtype}.")
            if not self.enable_kv_prewarming:
                logger.warning("CacheEngine: Assume transformers layers are equally partitioned by pipeline workers.")
                num_ele_per_layer = 1
                for size in kv_cache_shape:
                    num_ele_per_layer *= size
                start_layer = rank * self.num_attention_layers
                fill_compl = False
                num_layer = input_kv_caches.numel() // num_ele_per_layer
                if num_layer > start_layer:
                    for layer_idx in range(max(start_layer, 0), num_layer):
                        start_idx = layer_idx * num_ele_per_layer
                        kv_cache.append(input_kv_caches[start_idx:start_idx+num_ele_per_layer].view(kv_cache_shape))
                        if len(kv_cache) == self.num_attention_layers:
                            fill_compl = True
                            break
                if fill_compl:
                    return kv_cache
            else:
                # Note that a KV block contains all layers
                # Original block shape: kv_cache = [[2, num_blocks, block_size, num_kv_heads, head_size] for each layer]
                # We use the block shape of [2, num_blocks, num_layers * block_size, num_kv_heads, head_size]
                kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                    num_blocks * self.num_attention_layers, self.block_size, self.num_kv_heads, self.head_size)
                kv_cache_total_size = 2 * num_blocks * self.num_attention_layers * self.block_size * self.num_kv_heads * self.head_size
                assert kv_cache_total_size <= input_kv_caches.numel(), "Provided KV cache must have enough space."
                kv_cache = [input_kv_caches[:kv_cache_total_size].view(kv_cache_shape)]
                return kv_cache

        for _ in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            kv_cache.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        if self.enable_kv_prewarming:
            self.attn_backend.swap_blocks(self.cpu_cache[0], self.gpu_cache[0],
                                          src_to_dst)
        else:
            for i in range(self.num_attention_layers):
                self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                            src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        if self.enable_kv_prewarming:
            self.attn_backend.swap_blocks(self.gpu_cache[0], self.cpu_cache[0],
                                            src_to_dst)
        else:
            for i in range(self.num_attention_layers):
                self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                            src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_attention_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total
