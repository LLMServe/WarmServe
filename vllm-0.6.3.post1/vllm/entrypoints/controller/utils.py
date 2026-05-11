from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from dataclasses import dataclass

import os
import gc
import time
import ctypes
import torch
import socket
import functools
import traceback
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from multiprocessing.shared_memory import SharedMemory

from vllm.logger import init_logger
from vllm.config import LoadConfig
from vllm.executor.ray_utils import RayWorkerWrapper, ray
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
from vllm.utils import is_pin_memory_available
from vllm.distributed import init_distributed_environment, ensure_model_parallel_initialized
from vllm.distributed.parallel_state import change_default_world, reset_world

from .vmm import CUDAVMMPool
from .common import ModelInfo
from .memory_utils import get_shm_name, wrap_ptr_to_tensor
import cuda_allocator

logger = init_logger(__name__)

INIT_COMM_TIMEOUT = timedelta(seconds=5)

def catch_exceptions(func):
    """Catch exceptions for functions"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            error_msg = f"Method {func.__name__} meets exception: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Exception Type: {type(e).__name__}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return {"error": error_msg, "exception_type": type(e).__name__}
    return wrapper

class MemoryManager:
    def __init__(self, free_gpu_memory, device_id):
        stime = time.time()
        self.pool = CUDAVMMPool()
        self.pool_executor = ThreadPoolExecutor(max_workers=1)
        self.block_size = 2 * (2**20)   # 2MB
        self.granularity_ele = self.block_size // 2
        num_ele = free_gpu_memory // 2
        free_gpu_memory = num_ele * 2
        self.num_blocks = self.pool.initialize(free_gpu_memory / (2**30), 2, device_id)
        self.num_free_blocks = self.num_blocks
        self.phy_blocks = np.ones(self.num_blocks, dtype=bool)
        # Create prewarm slots. Each prewarm model uses one slot.
        self.max_prewarm_model = 30
        self.tensors = self.pool.init_virtual_tensors(self.max_prewarm_model)
        self.prewarm_slots = np.ones(self.max_prewarm_model, dtype=bool)
        self.load_streams = [torch.cuda.Stream(priority=-1) for _ in range(self.max_prewarm_model)]
        self.phy_block_ids = [np.zeros(0, dtype=int) for _ in range(self.max_prewarm_model)]
        self.released_block_ids = [np.zeros(0, dtype=int) for _ in range(self.max_prewarm_model)]
        etime_1 = time.time()

        # Allocate a CPU memory region
        num_ele_cpu = int(4 * (2 ** 30) // 2)   # assume use the default 4GB swap space
        self.cpu_mem = torch.zeros(
            num_ele_cpu,
            dtype=torch.float16,
            device=torch.device("cpu"),
            pin_memory=False,   # TODO(fix this): if pin_memory=True, a CUDA error will be raised
        )
        torch.cuda.synchronize()
        etime_2 = time.time()
        logger.info(f"Create GPU memory pool time cost = {'%.1f' % (etime_1 - stime)} s, CPU memory pool time cost = {'%.1f' % (etime_2 - etime_1)} s")
    
    def get_free_memory(self):
        return self.num_free_blocks * self.block_size

    def get_tensor(self, slot: int):
        # Get tensor of specific slot
        return self.tensors[slot]

    def reset(self):
        # free all prewarm slots
        available_slots = np.where(self.prewarm_slots)[0]
        for slot in available_slots:
            self.free_prewarm_slot(slot)
        return self.num_blocks * self.block_size

    def allocate_prewarm_slot(self):
        available_slots = np.where(self.prewarm_slots)[0]
        if len(available_slots) == 0:
            raise RuntimeError("No enough prewarm slots.")
        slot = available_slots[0]
        self.prewarm_slots[slot] = False
        return slot

    def free_prewarm_slot(self, slot: int):
        num_blocks = len(self.phy_block_ids[slot])
        num_released_blocks = len(self.released_block_ids[slot])
        # Async: Unmap blocks in background
        self.pool_executor.submit(self.pool.unmap_blocks, slot, 0, num_blocks * self.block_size)
        phy_block_ids_to_free = np.delete(self.phy_block_ids[slot], self.released_block_ids[slot])
        self.phy_blocks[phy_block_ids_to_free] = True
        self.phy_block_ids[slot] = np.zeros(0, dtype=int)
        self.released_block_ids[slot] = np.zeros(0, dtype=int)
        # NOTE: we can free this slot now because pool_executor executes jobs in strict order
        self.prewarm_slots[slot] = True
        self.num_free_blocks += num_blocks - num_released_blocks
        logger.info(f"Free Prewarm Slot: slot {slot} freed {num_blocks - num_released_blocks} blocks.")
    
    """
    Free the KV cache space of a slot
    """
    def free_kv_space(self, slot: int, kv_offset: int):
        kv_start_pos = kv_offset // self.block_size
        free_phy_block_ids = self.phy_block_ids[slot][kv_start_pos:]
        num_blocks = len(free_phy_block_ids)
        num_released_blocks = len(self.released_block_ids[slot])
        self.pool_executor.submit(self.pool.unmap_blocks, slot, kv_offset, num_blocks * self.block_size)
        phy_block_ids_to_free = np.delete(free_phy_block_ids, self.released_block_ids[slot] - kv_start_pos)
        self.phy_blocks[phy_block_ids_to_free] = True
        self.phy_block_ids[slot] = self.phy_block_ids[slot][:kv_start_pos]
        self.released_block_ids[slot] = np.zeros(0, dtype=int)
        self.num_free_blocks += num_blocks - num_released_blocks
        logger.info(f"Free KV Space: slot {slot} freed {num_blocks - num_released_blocks} blocks.")

    def allocate_blocks(self, slot: int, size: int = 0, sync_signal: List[int] = None, kv: bool = False, slow_event = None):
        if not size:
            # Allocate all free blocks
            num_blocks = self.num_free_blocks
        else:
            num_blocks = (size - 1) // self.block_size + 1
            if self.num_free_blocks < num_blocks:
                raise ValueError(f"No enough memory for allocating {size} bytes.")
        allocated_blocks = np.where(self.phy_blocks)[0][:num_blocks]

        # Append newly allocated blocks to this slot
        cur_offset = len(self.phy_block_ids[slot]) * self.block_size
        # Async: Map blocks in background
        self.pool_executor.submit(self.pool.map_blocks, slot, cur_offset, num_blocks, allocated_blocks, sync_signal, kv, slow_event)
        self.phy_block_ids[slot] = np.concatenate([self.phy_block_ids[slot], allocated_blocks])
        self.phy_blocks[allocated_blocks] = False
        self.num_free_blocks -= num_blocks
        logger.info(f"Allocate blocks: slot {slot} allocated {num_blocks} blocks.")
        return cur_offset, num_blocks * self.block_size

def get_weights(model_name_unify, rank, memory_model_loader, buffer_ptr):
    num_ele = 0
    weights = []
    if memory_model_loader:
        # load from memory
        buffer_size, state_dict = ray.get(memory_model_loader.get_dict.remote(model_name_unify, rank))
        for name, tensor_info in state_dict.items():
            offset, shape = tensor_info
            # get size
            size = 1
            for ele in shape:
                size *= ele
            tensor = wrap_ptr_to_tensor(buffer_ptr + offset, size * 2).view(shape)
            num_ele += size
            weights.append((name, tensor))
    else:
        # load from disk file
        file_name = os.path.join(
            model_name_unify,
            f"model-rank-{rank}-part-0.safetensors"
        )
        weights_iterator = safetensors_weights_iterator([file_name])

        for name, tensor in weights_iterator:
            num_ele += tensor.numel()
            weights.append((name, tensor))
    return num_ele, weights


class PrewarmModel:
    def __init__(self, model_info: ModelInfo, worker_ids: Tuple[int], world_id: str):
        self.model_info = model_info
        self.slot = None
        self.lock = threading.Lock()    # Lock operations on self.slot
        self.kv_offset = None
        self.worlds = {hash(worker_ids): world_id}
        self.model_loaded = {world_id: False}
        self.world_used = set()     # used worlds
    
    def add_world(self, worker_ids: Tuple[int], world_id: str):
        key = hash(worker_ids)
        exist = True if key in self.worlds else False
        self.worlds[key] = world_id
        self.model_loaded[world_id] = False
        return exist
    
    def get_world_id(self, worker_ids: Tuple[int]):
        hash_value = hash(worker_ids)
        if hash_value in self.worlds:
            return self.worlds[hash_value]
        return None

    def load_model(self, model_loader, memory_manager, in_background: bool, memory_model_loader, buffer_ptr, stop_event):
        stime = time.time()
        rank = self.model_info.rank
        pos = self.model_info.model_name.rfind('/')
        model_name_unify = self.model_info.model_name[:pos]

        num_ele, weights = get_weights(model_name_unify, rank, memory_model_loader, buffer_ptr)

        # Allocate a prewarm slot
        self.lock.acquire()
        if stop_event.is_set():
            self.lock.release()
            return
        self.slot = memory_manager.allocate_prewarm_slot()

        # Allocate space
        # We use a sync signal here to obtain how many blocks have been mapped 
        sync_signal = [0]
        offset, allocated_size = memory_manager.allocate_blocks(self.slot, num_ele * 2, sync_signal)
        self.lock.release()
        assert offset == 0, "Model weights should be placed at the beginning."
        self.kv_offset = allocated_size
        weight_tensor = memory_manager.get_tensor(self.slot)

        if stop_event.is_set():
            return

        # Perform tensor copy
        self.loaded_tensors = {}
        stream = memory_manager.load_streams[self.slot]
        num_ele = 0
        with torch.cuda.stream(stream):
            for name, tensor in weights:
                need_ele = tensor.numel()
                while (num_ele+need_ele) * 2 > sync_signal[0]:
                    time.sleep(0.01)
                if stop_event.is_set():
                    return
                tensor_ = weight_tensor[num_ele:num_ele+need_ele].view(tensor.shape)
                tensor_.copy_(tensor, non_blocking=True)
                self.loaded_tensors[name] = tensor_
                num_ele += need_ele
        if stop_event.is_set():
            return
        stream.synchronize()

        logger.info(f"Load model {self.model_info.model_name} rank {rank} time cost = {'%.2f' % (time.time() - stime)} seconds")
    
    """
    Destroy the resources of this class
    """
    def destroy(self, memory_manager):
        self.lock.acquire()
        if self.slot is not None:
            memory_manager.free_prewarm_slot(self.slot)
        self.lock.release()
        self.loaded_tensors = None
        for world_id in self.world_used:
            reset_world(world_id)
        logger.info(f"Free prewarmed model {self.model_info.model_name} with {len(self.world_used)} used worlds. Slot = {self.slot}.")

@catch_exceptions
def perform_model_load(self, need_load, model_name_unify, prewarm_model, world_id, in_background, stop_event, stop_comm_event, set_ctx_func, ctx):
    if stop_event.is_set():
        return
    stime = time.time()
    set_ctx_func(ctx)
    model_info = prewarm_model.model_info
    rank = model_info.rank
    if need_load:
        prewarm_model.load_model(self.prewarm_model_loader, self.memory_manager, in_background, self.memory_model_loader, self.data_ptr[(model_name_unify, rank)] if self.memory_model_loader else None, stop_event)
        if stop_event.is_set() or stop_comm_event.is_set():
            return
    # Create distributed environment
    # NOTE: We should obtain the port used by all workers here since we cannot determine the port early due to unknown start time of this function.
    my_file_name = f'port-{world_id}-{rank}.out'
    if rank == 0:
        # Obtain a port and write into the file
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', 0))
        master_port = sock.getsockname()[1]
        with open(my_file_name, 'w') as f:
            f.write(str(master_port))
            f.flush()
        sock.close()
    else:
        # Print "ready" information
        with open(my_file_name, 'w') as f:
            f.write("ready")
            f.flush()
    # Waiting for other ranks
    for peer_rank in range(model_info.world_size):
        if peer_rank != rank:
            file_name = f'port-{world_id}-{peer_rank}.out'
            while True:
                if os.path.exists(file_name):
                    with open(file_name, 'r') as f:
                        content = f.read()
                        if peer_rank == 0:
                            content = content.strip()
                            if content:
                                master_port = int(content)
                                break
                        elif content == "ready":
                            break
                if stop_event.is_set() or stop_comm_event.is_set():
                    return
    while True:
        try:
            init_distributed_environment(model_info.world_size,
                                        rank,
                                        f"tcp://127.0.0.1:{master_port}",
                                        self.local_rank,
                                        "nccl",
                                        world_id,
                                        INIT_COMM_TIMEOUT)
            break
        except Exception as e:
            # Timeout. Maybe due to peer stop loading. Check stop_event.
            logger.debug(f"Rank {self.local_rank} meets exception in init_distributed_environment: {e}.")
            if stop_event.is_set() or stop_comm_event.is_set():
                return
            time.sleep(0.05)
    ensure_model_parallel_initialized(model_info.tensor_parallel_size, model_info.pipeline_parallel_size, world_name=world_id)
    prewarm_model.model_loaded[world_id] = True
    logger.info(f"Rank {self.local_rank} prewarmed model {model_info.model_name} rank {rank} on slot {prewarm_model.slot}. Time cost = {'%.2f' % (time.time() - stime)} seconds")

class MyWorkerWrapper(RayWorkerWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_libraries()
        self.model_info = None
        self.worker_ids = None
        self.distributed_init_method = None
        self.world_id = None
        self.use_unified_memory = False
        self.memory_model_loader = None
        self.kv_cache_sync_signal = None
        self.kv_slow_event = threading.Event()
        self.kv_size = None
        self.last_checked_block_id = None
        self.kv_mapped = True
        self.num_gpu_blocks = None
        self.kv_block_size = None
        self.load_threads = {}      # hash(model_name, worker_ids) -> (model_info, stop_event, stop_comm_event, thread_handler)
        self.pool_executor = ThreadPoolExecutor(max_workers=1)
    
    def _init_libraries(self):
        from vllm.attention.backends.flash_attn import (  # noqa: F401
            FlashAttentionBackend)
        import vllm.worker.worker
        from vllm.model_executor.models.registry import ModelRegistry
        ModelRegistry.resolve_model_cls("LlamaForCausalLM")
    
    """
    Clear worker resources so that it can be allocated to another engine
    """
    @catch_exceptions
    def reset_worker(self, reset_prewarmed_model: bool):
        # reset worker status
        assert self.worker is not None
        self.worker = None
        self.model_info = None
        self.distributed_init_method = None
        self.world_id = None
        self.kv_cache_sync_signal = None
        self.kv_slow_event.clear()
        self.kv_size = None
        self.kv_mapped = True
        self.last_checked_block_id = None
        self.num_gpu_blocks = None
        self.kv_block_size = None
        if self.running_prewarm_model:
            if not reset_prewarmed_model:
                # Clear the KV Cache mapping
                self.memory_manager.free_kv_space(self.running_prewarm_model.slot, self.running_prewarm_model.kv_offset)
            self.running_prewarm_model = None
        if reset_prewarmed_model:
            if self.use_unified_memory:
                for prewarm_model in self.prewarm_models:
                    prewarm_model.destroy(self.memory_manager)
                self.prewarm_models = {}
            else:
                reset_world()
            if self.load_threads:
                stime = time.time()
                threads = []
                for key, (model_info, stop_event, stop_comm_event, thread_handler) in self.load_threads.items():
                    if not thread_handler.done():
                        stop_event.set()
                        threads.append(thread_handler)
                for thread_handler in threads:
                    thread_handler.result()
                self.load_threads = {}
                if threads:
                    logger.info(f"Rank {self.local_rank} stop {len(threads)} loading threads time cost = {'%.2f' % (time.time() - stime)} second")
        # clear resources
        os.environ.pop('VLLM_INSTANCE_ID', None)
        gc.collect()
        torch.cuda.empty_cache()
        if self.use_unified_memory:
            if reset_prewarmed_model:
                return self.memory_manager.reset()
            else:
                return self.memory_manager.get_free_memory()
        else:
            return self.get_free_gpu_memory()
    
    def prewarm_device(self, local_rank: int):
        # Initialize device
        os.environ['NCCL_CUMEM_ENABLE'] = '0'
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        gc.collect()
        torch.cuda.empty_cache()
        return self.get_free_gpu_memory()

    def get_total_gpu_memory(self):
        return self.total_gpu_memory
    
    def get_free_gpu_memory(self):
        self.free_gpu_memory = torch.cuda.mem_get_info()[0]
        return self.free_gpu_memory
    
    @catch_exceptions
    def update_model_info(self, model_info: ModelInfo, worker_ids: Tuple[int], distributed_init_method: Optional[str] = None, world_id: Optional[str] = None):
        self.model_info = model_info
        self.worker_ids = worker_ids
        if distributed_init_method:
            self.distributed_init_method = distributed_init_method
            self.world_id = world_id
        # Stop existing prewarming threads
        # NOTE: No need to wait util thread completion. This is overlapped with other initialization steps of workers and we will check completion before determine_num_available_blocks.
        if self.load_threads:
            new_key = hash((model_info.model_name, worker_ids))
            for key, (model_info, stop_event, stop_comm_event, thread_handler) in self.load_threads.items():
                if not thread_handler.done() and key != new_key:
                    if model_info != self.model_info:
                        stop_event.set()
                    else:
                        # Loading is for the same model but different workers. Just stop creating communication groups
                        stop_comm_event.set()
        # Free prewarmed models other than the needed one
        prewarm_models_ = {}
        for model_info, prewarm_model in self.prewarm_models.items():
            if model_info != self.model_info:
                prewarm_model.destroy(self.memory_manager)
            else:
                prewarm_models_[model_info] = prewarm_model
        self.prewarm_models = prewarm_models_
    
    def init_prewarm_loader(self, load_config: LoadConfig):
        self.prewarm_model_loader = get_model_loader(load_config)
        self.prewarm_models = {}
        self.running_prewarm_model = None
    
    def init_unified_memory(self, gpu_memory_utilization: float, enable_kv_prewarm: bool):
        free_gpu_memory = int(self.get_free_gpu_memory() * gpu_memory_utilization)
        self.memory_manager = MemoryManager(free_gpu_memory, self.local_rank)
        self.cuda_ctx = self.memory_manager.pool.get_current_context()
        self.use_unified_memory = True
        self.enable_kv_prewarm = enable_kv_prewarm
        return free_gpu_memory

    def init_model_loader(self, model_list: Dict[str, Dict[str, Tuple[int, int, int, str]]]):
        node_ip = self.get_node_ip()
        self.memory_model_loader = ray.get_actor(f"memory_model_manager_{node_ip}", namespace="prewarm")
        self.shms = []
        self.data_ptr = {}

        models = set()
        for task_type, model_dict in model_list.items():
            for model_id, model_info in model_dict.items():
                if model_id not in models:
                    models.add((model_id, model_info[1]))

        # Pin memory and obtain CUDA data_ptr
        for model_name, tp_size in models:
            for tp_rank in range(tp_size):
                buffer_size, state_dict = ray.get(self.memory_model_loader.get_dict.remote(model_name, tp_rank))
                shm_name = get_shm_name(model_name, tp_rank)
                shm = SharedMemory(name=shm_name, create=False, size=buffer_size)
                buffer_ptr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
                addr = cuda_allocator.register_pinned_memory(buffer_ptr, buffer_size)
                self.shms.append(shm)
                self.data_ptr[(model_name, tp_rank)] = addr

    """
    Release kv blocks for running instance so that these blocks can be used for prewarming other model.
    """
    @catch_exceptions
    def release_blocks(self, logical_block_indices: np.array, logical_block_size: int, total_logical_blocks: int):
        stime = time.time()
        # We use the size of each K block
        logical_block_size //= 2
        prewarm_model = self.running_prewarm_model
        physical_block_size = self.memory_manager.block_size
        num_phy_blocks_per_logical = logical_block_size // physical_block_size
        kv_delta = total_logical_blocks * num_phy_blocks_per_logical

        # Calculate all block indices that need to release
        base_indices = logical_block_indices[:, None] * num_phy_blocks_per_logical
        offsets = np.arange(num_phy_blocks_per_logical)[None, :]
        all_indices = (base_indices + offsets).ravel()
        k_indices = all_indices
        v_indices = all_indices + kv_delta

        # Release these blocks
        phy_block_ids = self.memory_manager.phy_block_ids[prewarm_model.slot]
        kv_block_ids = phy_block_ids[(prewarm_model.kv_offset//physical_block_size):]
        self.memory_manager.phy_blocks[kv_block_ids[k_indices]] = True
        self.memory_manager.phy_blocks[kv_block_ids[v_indices]] = True
        
        # Add these blocks into released_block_ids
        deleted_block_idx = np.concatenate([k_indices, v_indices])
        deleted_block_idx += prewarm_model.kv_offset // physical_block_size
        self.memory_manager.released_block_ids[prewarm_model.slot] = np.concatenate([self.memory_manager.released_block_ids[prewarm_model.slot], deleted_block_idx])

        num_released_blocks = len(logical_block_indices) * num_phy_blocks_per_logical * 2
        self.memory_manager.num_free_blocks += num_released_blocks
        logger.info(f"Rank {self.local_rank} released {num_released_blocks} blocks.")
        logger.info(f"Rank {self.local_rank} release_blocks time cost = {'%1f' % (time.time() - stime)} seconds")

    """
    Load a prewarming model according to model_info
    """
    @catch_exceptions
    def load_model(self, model_info: ModelInfo, worker_ids: Tuple[int], world_id: str, in_background: bool = False) -> PrewarmModel:
        stime = time.time()
        if not self.prewarm_model_loader:
            raise RuntimeError("You need to first initialize model loader.")
        pos = model_info.model_name.rfind('/')
        model_name_unify = model_info.model_name[:pos]
        if model_info not in self.prewarm_models:
            prewarm_model = PrewarmModel(model_info, worker_ids, world_id)
            self.prewarm_models[model_info] = prewarm_model
            need_load = True
        else:
            # New workers with the same model. Just add a new communication group.
            prewarm_model = self.prewarm_models[model_info]
            exist = prewarm_model.add_world(worker_ids, world_id)
            if exist:
                # The same worker_ids have tried to establish connection. Stop the previous connection establishing process because we are going to use the new world_id.
                # NOTE: this happens when the peer first stop prewarming this model, and after a while begins to prewarm this model again.
                for key, (load_model_info, stop_event, stop_comm_event, thread_handler) in self.load_threads.items():
                    if not thread_handler.done():
                        if load_model_info == model_info:
                            stop_comm_event.set()
            need_load = False

        stop_event = threading.Event()
        stop_comm_event = threading.Event()
        load_thread = self.pool_executor.submit(perform_model_load, self, need_load, model_name_unify, prewarm_model, world_id, in_background, stop_event, stop_comm_event, self.memory_manager.pool.set_current_context, self.cuda_ctx)
        self.load_threads[hash((model_info.model_name, worker_ids))] = (model_info, stop_event, stop_comm_event, load_thread)
        logger.info(f"Rank {self.local_rank} start load model {model_info.model_name} time cost = {'%1f' % (time.time() - stime)} seconds")
    
    @catch_exceptions
    def check_prewarm_compl(self, model_info: ModelInfo, worker_ids: Tuple[int]):
        if model_info not in self.prewarm_models:
            raise ValueError("check_prewarm_compl: Model not loaded.")
        prewarm_model = self.prewarm_models[model_info]
        world_id = prewarm_model.get_world_id(worker_ids)
        if world_id is None:
            raise ValueError(f"check_prewarm_compl: World for {worker_ids} not found.")
        return prewarm_model.model_loaded[world_id]
    
    @catch_exceptions
    def free_model(self, model_info: ModelInfo):
        if model_info not in self.prewarm_models:
            raise ValueError("free_model: Model not loaded.")
        prewarm_model = self.prewarm_models[model_info]
        prewarm_model.destroy(self.memory_manager)
        del self.prewarm_models[model_info]
    
    """
    Overwrite execute_method to intercept function calls.
    """
    @catch_exceptions
    def execute_method(self, method, *args, **kwargs):
        if method == "init_device":
            if self.model_info in self.prewarm_models:
                prewarm_model = self.prewarm_models[self.model_info]
                world_id = prewarm_model.get_world_id(self.worker_ids)
                if world_id is None:
                    raise RuntimeError(f"Rank {self.local_rank}: World for {self.worker_ids} not found.")
                while not prewarm_model.model_loaded[world_id]:
                    time.sleep(0.05)
                logger.debug(f"Rank {self.local_rank} change default world to {world_id}")
                change_default_world(world_id)
                prewarm_model.world_used.add(world_id)
                self.running_prewarm_model = prewarm_model
                kwargs.update(skip_set_device=True)
            elif self.use_unified_memory:
                raise RuntimeError("If unified memory enabled, only prewarmed model is allowed.")
            else:
                # Normal model initialization
                world_id = self.world_id
                init_distributed_environment(self.model_info.world_size,
                                self.model_info.rank,
                                self.distributed_init_method,
                                self.local_rank,
                                "nccl",
                                world_id)
                ensure_model_parallel_initialized(self.model_info.tensor_parallel_size, self.model_info.pipeline_parallel_size, world_name=world_id)
                logger.debug(f"Rank {self.local_rank} change default world to {world_id}")
                change_default_world(world_id)
                kwargs.update(skip_set_device=True)
        elif method == "load_model":
            if self.model_info in self.prewarm_models:
                prewarm_model = self.prewarm_models[self.model_info]
                kwargs.update(loaded_tensors=prewarm_model.loaded_tensors)
            elif self.memory_model_loader:
                # Prewarming not enabled and load model from memory.
                rank = self.model_info.rank
                pos = self.model_info.model_name.rfind('/')
                model_name_unify = self.model_info.model_name[:pos]
                num_ele, weights = get_weights(model_name_unify, rank, self.memory_model_loader, self.data_ptr[(model_name_unify, rank)])
                loaded_tensors = {}
                for name, tensor in weights:
                    loaded_tensors[name] = tensor.to(self.device)
                kwargs.update(loaded_tensors=loaded_tensors)
            logger.debug(f"Rank {self.local_rank} finish load_model.")
        elif method == "determine_num_available_blocks":
            if self.model_info in self.prewarm_models:
                # Before get_free_memory, ensure that all existing load threads have completed.
                if self.load_threads:
                    stime = time.time()
                    for key, (model_info, stop_event, stop_comm_event, thread_handler) in self.load_threads.items():
                        if not thread_handler.done():
                            thread_handler.result()
                    self.load_threads = {}
                    logger.debug(f"Rank {self.local_rank} waiting for load threads completion time cost = {'%.2f' % (time.time() - stime)} seconds.")
                free_gpu_memory = self.memory_manager.get_free_memory()
                kwargs.update(free_gpu_memory=free_gpu_memory)
        elif method == "initialize_cache":
            if self.model_info in self.prewarm_models:
                # NOTE: we will later check kv cache mapping
                prewarm_model = self.prewarm_models[self.model_info]
                self.kv_size = self.num_gpu_blocks * self.kv_block_size
                self.kv_cache_sync_signal = [0]
                self.last_checked_block_id = -1
                logger.debug(f"Rank {self.local_rank}: initialize_cache for size = {self.kv_size}")
                offset, allocated_size = self.memory_manager.allocate_blocks(prewarm_model.slot, self.kv_size, self.kv_cache_sync_signal, kv=self.enable_kv_prewarm, slow_event=self.kv_slow_event)
                gpu_block = self.memory_manager.get_tensor(prewarm_model.slot)[offset//2:(offset+allocated_size)//2]
                cpu_block = self.memory_manager.cpu_mem
                self.kv_mapped = False
                kwargs.update(kv_caches=(gpu_block, cpu_block))
        elif method == "init_worker":
            kwargs.update(local_rank=self.local_rank)
        
        ret = super().execute_method(method, *args, **kwargs)

        if method == "determine_num_available_blocks":
            if self.model_info in self.prewarm_models:
                self.num_gpu_blocks = ret[0]
                self.kv_block_size = ret[2]
                ret = (ret[0], ret[1])
                logger.debug(f"Rank {self.local_rank}: num_gpu_blocks = {self.num_gpu_blocks}, size_per_block = {self.kv_block_size}")

        return ret
    
    def execute_model_spmd(
            self, req_or_tuple
        ) -> bytes:
        if not self.kv_mapped:
            if not self.enable_kv_prewarm:
                stime = time.time()
                while self.kv_cache_sync_signal[0] < self.kv_size:
                    time.sleep(0.05)
                self.kv_mapped = True
                logger.debug(f"Rank {self.local_rank} KV mapped")
                logger.debug(f"Rank {self.local_rank} waiting for KV cache mapping time cost = {'%.2f' % ((time.time() - stime) * 1000)} ms")
                return super().execute_model_spmd(req_or_tuple)
            
            if self.kv_cache_sync_signal[0] >= self.kv_size:
                # All KV cache has mapped
                self.kv_mapped = True
                logger.debug(f"Rank {self.local_rank} KV mapped")
                return super().execute_model_spmd(req_or_tuple)
            # Check whether the mapped region can serve current requests
            # Decode the request to obtain how many tokens to process
            stime = time.time()
            if isinstance(req_or_tuple, bytes):
                serialized_req, intermediate_tensors = req_or_tuple, None
                execute_model_req = self.input_decoder.decode(serialized_req)
            elif isinstance(req_or_tuple, tuple):
                serialized_req, intermediate_tensors = req_or_tuple
                for key, value in intermediate_tensors.tensors.items():
                    intermediate_tensors.tensors[key] = value.cuda()
                execute_model_req = self.input_decoder.decode(serialized_req)
            
            mx_block_id = -1
            for seq_group_metadata in execute_model_req.seq_group_metadata_list:
                for block_list in seq_group_metadata.block_tables.values():
                    mx_block_id = max(mx_block_id, max(block_list, default=-1))
            
            if mx_block_id <= self.last_checked_block_id:
                # No need to check this block id
                return super().execute_model_spmd((serialized_req, intermediate_tensors, execute_model_req))
            self.kv_slow_event.clear()      # Clear slow flag that might be set previously
            self.last_checked_block_id = mx_block_id
            required_mapping_pos = (mx_block_id + 1) * self.kv_block_size // 2    # Required K cache position that has been mapped
            logger.debug(f"Rank {self.local_rank} maximum block id = {mx_block_id}")
        
            # Check whether KV cache has mapped to our required position
            while self.kv_cache_sync_signal[0] < required_mapping_pos:
                time.sleep(0.01)
            self.kv_slow_event.set()        # Set slow flag to avoid interference with inference
            logger.debug(f"Rank {self.local_rank} waiting for KV cache mapping time cost = {'%.2f' % ((time.time() - stime) * 1000)} ms")
            return super().execute_model_spmd((serialized_req, intermediate_tensors, execute_model_req))
        
        return super().execute_model_spmd(req_or_tuple)