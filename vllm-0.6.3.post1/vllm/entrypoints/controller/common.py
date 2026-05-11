from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from dataclasses import dataclass
from collections import deque
import struct
from multiprocessing.shared_memory import SharedMemory

'''
Batch Size
'''
BATCH_SIZE = 32

'''
Autoscaling Metrics
'''
SCALE_LOWER_BOUND = 0.5
SCALE_UPPER_BOUND = 0.8
PROTECT_PERIOD = 5
SCALE_WINDOW = 5

'''
Load Speed (GB/s) (to estimate cold start latency)
'''
PCIE_BANDWIDTH = 32

'''
GPU Memory Granularity
'''
MEM_GRANULARITY = 2 ** 21

def align_up(size_in_bytes: int):
    # Align up to 2MB
    return ((size_in_bytes - 1) // MEM_GRANULARITY + 1) * MEM_GRANULARITY
@dataclass(frozen=True)
class ModelInfo:
    model_name: str = ""
    tensor_parallel_rank: int = -1
    tensor_parallel_size: int = 1
    pipeline_parallel_rank: int = -1
    pipeline_parallel_size: int = 1

    @property
    def rank(self):
        return self.pipeline_parallel_rank * self.tensor_parallel_size + self.tensor_parallel_rank
    
    @property
    def world_size(self):
        return self.tensor_parallel_size * self.pipeline_parallel_size

    # Get ModelInfo for a specific rank
    def get_info(self, rank: int):
        pp_rank = rank // self.tensor_parallel_size
        tp_rank = rank % self.tensor_parallel_size
        return ModelInfo(self.model_name, tp_rank, self.tensor_parallel_size, pp_rank, self.pipeline_parallel_size)

@dataclass
class WorkerStatus:
    # The node of the worker
    node: int = -1
    # The GPU id of the worker
    gpu_id: int = 0
    # If worker has not been allocated, then $model is None
    model: Optional[ModelInfo] = None
    # The prewarmed models that the worker holds
    prewarm_models: Optional[List[ModelInfo]] = None

    def __post__init__(self):
        if self.prewarm_models is None:
            self.prewarm_models = []

@dataclass
class EngineStatus:
    # Engine id
    id: int = -1
    # Model name
    model: str = ""
    # Workers
    workers: Optional[Tuple[int]] = None
    # Current running request
    num_reqs: int = 0
    # Stopping
    stopping: bool = False
    # Prewarm engine id
    prewarm_engine_id: int = -1
    # Creation time
    time_created: Optional[float] = None

@dataclass
class ModelStatus:
    # Current running request
    num_reqs: int = 0
    # Sum of (current_load * elapse). For Prewarm.
    sum_loads: float = 0
    # Peak load. For Prewarm.
    max_loads: int = 0
    # Last recorded time. For Prewarm.
    last_time: Optional[float] = None
    # Stopping engines
    stopping_engines: Optional[Set[int]] = None
    # Past loads. Record the (time, cur_load) for autoscaler.
    load_stamp: Optional[deque[Tuple[float, int]]] = None

    def get_avg_load(self, cur_time: float):
        if len(self.load_stamp) == 0:
            return 0
        st_time = cur_time - SCALE_WINDOW
        if len(self.load_stamp) == 1:
            time, load = self.load_stamp[0]
            if time < st_time:
                return load
            return load * (cur_time - time) / SCALE_WINDOW
        while len(self.load_stamp) >= 2 and self.load_stamp[1][0] <= st_time:
            self.load_stamp.popleft()
        if len(self.load_stamp) == 1:
            # No data point in the window
            return self.load_stamp[0][1]
        if self.load_stamp[0][0] > st_time:
            st_pos = 0
            last_time, last_load = st_time, 0
        else:
            st_pos = 1
            last_time, last_load = self.load_stamp[0]
        sum_load = 0
        for i in range(st_pos, len(self.load_stamp)):
            sum_load += last_load * (self.load_stamp[i][0] - last_time)
            last_time, last_load = self.load_stamp[i]
        sum_load += last_load * (cur_time - last_time)
        return sum_load / SCALE_WINDOW

class EngineData:
    def __init__(self, prewarm_id: int):
        # KV block allocator
        self.total_blocks = None
        self.block_size = None
        # Records target free_blocks.
        self.kv_shm = SharedMemory(create=True, size=100000, name=f"kv-{prewarm_id}")
        self.kv_shm.buf[:4] = struct.pack("i", 1000000000)
        self.kv_shm.buf[4:8] = struct.pack("i", 0)
        self.last_reserved_blocks = 0
    
    def destroy(self):
        if self.kv_shm:
            self.kv_shm.close()
            # self.kv_shm.unlink()  # TODO(fix this): unlink leads to FileNotFoundError during engine shutdown
            self.kv_shm = None