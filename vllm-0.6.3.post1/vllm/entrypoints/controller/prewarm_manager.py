from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from collections import defaultdict

import os
import gc
import sys
import copy
import math
import time
import yaml
import pickle
import struct
import argparse
import subprocess
import numpy as np
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from vllm.config import LoadConfig, ParallelConfig
from vllm.executor.ray_utils import RayWorkerWrapper, ray
from vllm.worker.worker_base import WorkerBase
from vllm.executor.ray_utils import _wait_until_pg_ready
from vllm.utils import get_distributed_init_method, get_ip, get_open_port

from .utils import MyWorkerWrapper
from .ModelConfig import ModelKVConfig, ModelList
from .common import BATCH_SIZE, SCALE_LOWER_BOUND, SCALE_UPPER_BOUND, PROTECT_PERIOD
from .common import ModelInfo, WorkerStatus, ModelStatus, EngineStatus, EngineData
from .scheduler import Scheduler
from .muxserve_utils import create_muxserve_engine

import traceback
import fastapi
import uvicorn
import requests
from fastapi import Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
import asyncio

if ray is not None:
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val

"""
PrewarmManager is responsible of prewarming:
1. Device. [By creating workers in prior.]
2. Model. [By loading model to workers in prior.]
3. Engine. [By creating subprocess that imports libraries in prior.]
"""

@ray.remote(num_cpus=1)
class PrewarmManager:
    def __init__(self, args):
        self.node_id = ray.get_runtime_context().get_node_id()
        self.use_unified_memory = args.use_unified_memory
        self.disable_all_prewarm = args.disable_all_prewarm
        self.enable_prewarm = not args.disable_prewarm
        self.use_loaded_model = args.use_loaded_model
        if self.disable_all_prewarm and self.use_unified_memory:
            raise ValueError("Prewarm is disabled. Please do not enable unified memory.")
        self.enable_kv_prewarm = True if self.use_unified_memory and self.enable_prewarm and not args.disable_kv_prewarm else False
        self.disable_placement = args.disable_placement
        if self.disable_placement:
            print("Smart placement disabled.")
        self.default_gpu_memory_utilization = args.gpu_memory_utilization
        self.vllm_env = self._get_worker_env()

        self.is_muxserve = args.muxserve
        if self.is_muxserve:
            if not args.placement_config:
                raise ValueError("Placement configuration not provided.")
            base_placement_config = args.placement_config
            served_models = os.getenv("MODELS", "").split(",")
            engine_id = 0
            self.engines = []
            self.model_info = {}
            self.batch_size = {}
            self.engine_reqs = {}
            self.model_engines = defaultdict(set)
            cur_device_pos = 0
            for idx in range(4):
                exist = False
                for sz in [1,2,4,8]:
                    placement_config = f"{base_placement_config}{sz}_idx{idx}.yaml"
                    if os.path.exists(placement_config):
                        exist = True
                        break
                if not exist:
                    if idx == 0:
                        # try to directly add a ".yaml"
                        placement_config = base_placement_config + ".yaml"
                        if os.path.exists(placement_config):
                            exist = True
                    if not exist:
                        continue
                with open(placement_config, 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file)
                gpu_memory_utilization = config["gpu_memory_utilization"]
                tot_percent = 0
                print("Start model engines...")
                total_block_size = 0
                total_percent = 0
                has_served_model = False
                for model_info in config["models"]:
                    name = model_info["name"]
                    model_name = model_info["model"]
                    mps_percent = model_info["mps_percentage"]
                    mps_percent = max(mps_percent[0], mps_percent[1])
                    total_percent += mps_percent
                    block_size = None
                    for key, value in ModelKVConfig.items():
                        if model_name in key:
                            block_size = value[1]
                    total_block_size += block_size
                    if name in served_models:
                        has_served_model = True
                if not has_served_model:
                    continue
                for model_info in config["models"]:
                    name = model_info["name"]
                    if name not in served_models:
                        continue 
                    model_name = model_info["model"]
                    mps_percent = model_info["mps_percentage"]
                    # mps_percent = max(mps_percent[0], mps_percent[1])
                    mps_percent = max(mps_percent[0], mps_percent[1]) / total_percent * 100
                    max_num_seqs = model_info["max_num_seqs"]
                    tensor_parallel_size = model_info["tensor_parallel_size"]

                    tot_percent += mps_percent
                    # Launch a MuxServe Model
                    model_engine = create_muxserve_engine(model_name, tensor_parallel_size, cur_device_pos, mps_percent, gpu_memory_utilization*(block_size/total_block_size), max_num_seqs, engine_id)
                    self.batch_size[engine_id] = max_num_seqs
                    self.model_engines[model_name].add(engine_id)
                    self.engine_reqs[engine_id] = 0
                    self.model_info[model_name] = True
                    self.engines.append(model_engine)
                    engine_id += 1
                cur_device_pos += tensor_parallel_size
                if cur_device_pos > 8:
                    raise ValueError("Not enough device!")

            stime = time.time()
            for idx in range(engine_id):
                # Wait for server startup
                engine_addr = "http://localhost:" + str(5050+idx) + "/v1"
                while True:
                    try:
                        response = requests.get(engine_addr, timeout=1)
                        break
                    except requests.exceptions.RequestException as e:
                        pass
            print(f"Engine started. Time cost = {'%.2f' % (time.time() - stime)} seconds")
            return

        # ----- Workers -----
        # workers: List[worker_wrapper]
        # node_list: node name list
        # node_workers: List[worker_id list of each node]
        # worker_node_gpu_ids: List[(worker_node_id, worker_gpu_ids)]
        # gpu_mem: List[worker_gpu_memory_in_bytes]
        parts = filter(None, [x.strip() for x in args.disabled_gpus.split(',')])
        self.disabled_gpus = [int(x) for x in parts]
        if self.disable_all_prewarm:
            resources = ray.cluster_resources()
            self.num_workers = int(resources.get("GPU", 0))
            self.num_servers = len(ray.nodes())
            self.num_gpu_per_server = self.num_workers // self.num_servers
            self.node_workers = [[node_id * self.num_gpu_per_server + rank for rank in range(self.num_gpu_per_server)] for node_id in range(self.num_servers)]
            self.worker_node_gpu_ids = [(rank // self.num_gpu_per_server, [rank % self.num_gpu_per_server]) for rank in range(self.num_workers)]
            self.gpu_mem = [0 for _ in range(self.num_workers)]
            self.placement_group = self._initialize_ray_cluster()
        else:
            self.load_config = LoadConfig(
                load_format=args.load_format,
                download_dir=args.download_dir,
                model_loader_extra_config=args.model_loader_extra_config,
                ignore_patterns=args.ignore_patterns,
            )
            self.placement_group = self._initialize_ray_cluster()
            self.workers, self.node_list, self.node_workers, self.worker_node_gpu_ids, self.gpu_mem = self._init_workers_ray(self.placement_group, self.use_loaded_model)
            self.num_servers = len(self.node_list)
            self.num_workers = len(self.workers)
            self.num_gpu_per_server = self.num_workers // self.num_servers

        self.worker_stats: List[WorkerStatus] = []          # Locked by model_lock
        for (node_id, gpu_ids) in self.worker_node_gpu_ids:
            self.worker_stats.append(WorkerStatus(node=node_id, gpu_id=gpu_ids[0]))

        if self.use_unified_memory:
            stime = time.time()
            futures = []
            for worker in self.workers:
                futures.append(worker.init_unified_memory.remote(self.default_gpu_memory_utilization, self.enable_kv_prewarm))
            results = ray.get(futures)
            self.mem_per_gpu = results[0]
            self.gpu_mem = results      # Available GPU memory for prewarming
            print(f"Init unified memory of {'%.1f' % (self.mem_per_gpu * self.num_workers / (2**30))} GB complete. Time cost = {'%.1f' % (time.time() - stime)} seconds")

        # ----- Models -----
        self.model_info: Dict[str, Tuple[int, int, str]] = {}    # Dict[model -> (size_in_gb, tp, datasource)]
        # model_engines, model_stats, num_engines, num_in_creating_engines, and recorded_reqs are Locked by engine_lock
        self.model_engines: Dict[str, Set[int]] = defaultdict(set)      # Dict[model -> Set[Engines]]
        self.model_stats: Dict[str, ModelStatus] = {}                   # Model status that used to create or remove engines
        self.num_engines: Dict[str, int] = {}
        self.num_in_creating_engines: Dict[str, int] = {}
        self.recorded_reqs = set()
        model_cur_index = {}
        for task_type, model_dict in ModelList.items():
            for model_id, model_info_ in model_dict.items():
                if model_id not in model_cur_index:
                    model_cur_index[model_id] = 0
                    cur_index = 0
                else:
                    cur_index = model_cur_index[model_id]
                for index in range(model_info_[2]):
                    split_model_id = model_id + ("_mem" if self.disable_all_prewarm and self.use_loaded_model else "") + "/" + str(index + cur_index)
                    self.model_info[split_model_id] = (model_info_[0], model_info_[1], model_info_[3])
                    self.num_engines[split_model_id] = 0
                    self.num_in_creating_engines[split_model_id] = 0
                    self.model_stats[split_model_id] = ModelStatus(stopping_engines=set(), load_stamp=deque())
                model_cur_index[model_id] += model_info_[2]

        # model_workers: Dict[model -> List[Tuple[worker_id]]]
        #     For a prewarmed model, record the worker lists that host the model
        # node_models: List[Set[Tuple[model, Tuple[worker_id]]]]
        #     For each node, record the list of prewarmed models
        self.model_workers: Dict[str, Set[Tuple[int]]] = defaultdict(set)
        self.node_models: List[Set[Tuple[str, Tuple[int]]]] = [set() for _ in range(self.num_servers)]

        # in_prewarming_models: Dict[Node, List[Tuple[ModelInfo, workers, List[Futures]]]]
        self.world_counter = 0
        if self.use_unified_memory:
            self.in_prewarming_models = defaultdict(list)
            if self.enable_prewarm:
                self.prewarm_scheduler = Scheduler(self.model_info, self.num_servers, len(self.node_workers[0]), self.disable_placement, args.window_size)
                if not args.character_file:
                    raise ValueError("To enable prewarming, you should provide the path to characteristics file.")
                with open(args.character_file, 'rb') as f:
                    characters = pickle.load(f)
                self.prewarm_scheduler.init(characters)
                self.init_timer()
        
        # ----- Engines -----
        self.prewarm_engine_counter = 0
        self.max_concurrent_engines = 32
        self.engine_runout_threshold = 10

        # prewarmed_engines: List[Tuple[prewarm_id, engine_process]]
        # running_engines: Dict[prewarm_id, engine_process]
        os.system("ps aux | grep api_server | grep -v grep | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")      # kill previous engines
        self.running_engines = {}
        if not self.disable_all_prewarm:
            # TODO:(fix this) pre-initializa tokenizer causes error: huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
            # self.tokenizer_path = list(self.model_info.keys())[0]
            # self.vllm_env["TOKENIZER_PATH"] = self.tokenizer_path
            self.prewarmed_engines = self._init_engines(self.max_concurrent_engines)
        # We may have appended some model information to prewarmed engines during scheduling
        self.prewarm_engine_info = {}
        
        self.engine_counter = 0
        self.engine_stats: List[EngineStatus] = {}
        os.system("rm /dev/shm/kv-*")           # kill previous shared memory
        os.system("rm port-*.out")              # rm port-related files
        if self.enable_kv_prewarm:
            self.engine_data: List[EngineData] = [EngineData(i) for i in range(self.max_concurrent_engines)]

        self.prewarmed_engine_extend_future = None
        self.engine_data_extend_future = None
        self.engine_extend_executor = ThreadPoolExecutor(max_workers=1)
        
        # ----- Loops -----
        self.engine_lock = threading.Lock()     # Protect modifying engine-related information
        self.model_lock = threading.Lock()      # Protect modifying model-related information
        self.engine_loop = threading.Thread(target=self.check_engine_loop)
        self.engine_loop.start()
        if self.use_unified_memory:
            self.prewarm_loop = threading.Thread(target=self.prewarm_model_loop)
            self.prewarm_loop.start()
        self.print_loop = threading.Thread(target=self.print_loop)
        self.print_loop.start()
        
        if args.prewarm_model:
            # -----TESTING-----
            # Prewarm a model in prior
            stime = time.time()
            # prewarm model and kv cache
            tp_size = args.prewarm_model_tp_size
            pp_size = args.prewarm_model_pp_size
            world_size = tp_size * pp_size
            worker_ids = range(world_size)
            ret = self.prewarm_model_async(args.prewarm_model_name, tp_size, pp_size, worker_ids)
            self.model_lock.acquire()
            self.in_prewarming_models[0].append(ret)
            self.model_lock.release()
            while self.in_prewarming_models[0]:
                time.sleep(1)
            print(f"Prewarm model complete. Time cost = {time.time() - stime} seconds")

    def get_num_workers(self):
        if self.is_muxserve:
            return 0
        return self.num_workers

    """
    Allocate workers while engine is on $engine_node_id.
    """
    def get_workers(self, engine_node_id: str, parallel_config: ParallelConfig, model: str, prewarm_engine_id: int):
        if prewarm_engine_id in self.prewarm_engine_info:
            # The engine is created by our scheduler
            worker_ids, engine_id = self.prewarm_engine_info[prewarm_engine_id]
            return [self.workers[id] for id in worker_ids], engine_id

        # The engine is created by user
        model_info = ModelInfo(
            model, tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size)
        node_id, worker_ids, prewarm_hit = self._get_available_workers(model_info, engine_node_id, parallel_config.world_size)
        if not worker_ids:
            # No available workers
            return None, None

        if prewarm_hit:
            # Do not change the order of workers
            sorted_worker_ids = worker_ids
        else:
            # Sort workers
            node_count = {}
            for id in worker_ids:
                node_id = self.worker_stats[id].node
                node_count[node_id] = node_count.get(node_id, 0) + 1
            
            def sort_by_driver_then_worker_ip(map_id):
                """
                Sort the workers based on 3 properties:
                1. If the worker is on the same node as the driver (vllm engine),
                    it should be placed first.
                2. Then, if the worker is on a node with fewer workers, it should
                    be placed first.
                3. Finally, if the work is on a node with smaller IP address, it
                    should be placed first.
                """
                node_id = self.worker_stats[worker_ids[map_id]].node
                return (node_id != engine_node_id, node_count[node_id], node_id)
            mapping = sorted(list(range(len(worker_ids))), key=sort_by_driver_then_worker_ip)
            sorted_worker_ids = []
            for map_id in mapping:
                sorted_worker_ids.append(worker_ids[map_id])
        
        # Obtain model partition for each worker
        futures = []
        pp_rank = 0
        tp_rank = 0
        self.model_lock.acquire()   # Acquire model_lock to modify worker_stats and gpu_mem
        for worker_id in sorted_worker_ids:
            model_info = ModelInfo(
                model_name=model,
                tensor_parallel_rank=tp_rank,
                tensor_parallel_size=parallel_config.tensor_parallel_size,
                pipeline_parallel_rank=pp_rank,
                pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            )
            self.worker_stats[worker_id].model = model_info
            self.gpu_mem[worker_id] = 0

            # Update model information in worker wrapper
            # Note that the worker_ids has the same order with worker_ids for prewarmed model
            futures.append(self.workers[worker_id].update_model_info.remote(model_info, worker_ids))

            if tp_rank == parallel_config.tensor_parallel_size - 1:
                pp_rank += 1
                tp_rank = 0
            else:
                tp_rank += 1
        self.model_lock.release()
        
        # NOTE: no need to wait for update_model_info completion
        # ray.get(futures)
                
        # Register a new engine
        self.engine_lock.acquire()
        engine_id = self.engine_counter
        self.engine_counter += 1
        self.engine_stats[engine_id] = EngineStatus(id=engine_id, model=model, workers=sorted_worker_ids, prewarm_engine_id=prewarm_engine_id, time_created=time.time())
        self.model_engines[model].add(engine_id)
        self.num_engines[model] += 1

        # Check whether we have enough engine_data prepared
        if self.enable_kv_prewarm:
            if len(self.engine_data) - engine_id < self.engine_runout_threshold:
                if not self.engine_data_extend_future:
                    def extend_engine_data(start_idx: int, num_engine: int):
                        engine_data = []
                        for idx in range(start_idx, start_idx + num_engine):
                            engine_data.append(EngineData(idx))
                        return engine_data
                    self.engine_data_extend_future = self.engine_extend_executor.submit(extend_engine_data, len(self.engine_data), self.max_concurrent_engines)
                extended_engine_data = None
                if len(self.engine_data) <= engine_id:
                    print(f"Warning: No available engine data. Waiting for creating new data.")
                    stime = time.time()
                    extended_engine_data = self.engine_data_extend_future.result()
                    print(f"Waiting for data creation time cost = {'%.2f' % (time.time() - stime)} seconds")
                elif self.engine_data_extend_future.done():
                    extended_engine_data = self.engine_data_extend_future.result()
                if extended_engine_data:
                    self.engine_data.extend(extended_engine_data)
                    self.engine_data_extend_future = None
        self.engine_lock.release()

        return [self.workers[id] for id in sorted_worker_ids], engine_id

    """
    Register GPU KV block allocator.
    """
    def register_kv(self, prewarm_id: int, total_blocks):
        self.engine_data[prewarm_id].total_blocks = total_blocks
        model = self.engine_stats[prewarm_id].model
        if model not in ModelKVConfig:
            # remove suffix number
            pos = model.rfind('/')
            model = model[:pos]
        self.engine_data[prewarm_id].block_size = ModelKVConfig[model][1] * (2**20)

    def _get_worker_env(self):
        env = {
            "VLLM_USE_RAY_SPMD_WORKER": "1",
            "SKIP_RAY_VERIFY": "1",
            "VLLM_DO_NOT_TRACK": "1",
            # "VLLM_USE_RAY_COMPILED_DAG": "1",     # Do not use compiled dag because it requires calling dag.teardown() after engine shutdown, but we directly kill the RayGPUExecutor.
            "VLLM_WARMUP_DEVICE": "1" if not self.disable_all_prewarm else "0",
            "VLLM_PREWARM_MODEL": "1" if self.use_unified_memory else "0",
            "VLLM_PREWARM_ENGINE": "1",
            "VLLM_KV_PREWARMING": "1" if self.enable_kv_prewarm else "0",
            "VLLM_LOGGING_LEVEL": "DEBUG",
        }
        return env

    def _get_available_workers(self, model_info: ModelInfo, engine_node_id: str, world_size: int):
        self.model_lock.acquire()
        if self.use_unified_memory:
            # 1. Try to get prewarmed model
            if model_info.model_name in self.model_workers:
                for worker_list in self.model_workers[model_info.model_name]:
                    available = True
                    for worker_id in worker_list:
                        if self.worker_stats[worker_id].model is not None:
                            available = False
                            break
                    if available:
                        for rank, worker_id in enumerate(worker_list):
                            self.worker_stats[worker_id].model = model_info.get_info(rank)
                        self.model_lock.release()
                        return (worker_list[0] // self.num_gpu_per_server), worker_list, True
            # 2. Try to get in-prewarming model
            for node_id, in_prewarming_model_list in self.in_prewarming_models.items():
                for prewarm_model_info, worker_list, futures, unfin_workers in in_prewarming_model_list:
                    if prewarm_model_info == model_info:
                        available = True
                        for worker_id in worker_list:
                            if self.worker_stats[worker_id].model is not None:
                                available = False
                                break
                        if available:
                            for rank, worker_id in enumerate(worker_list):
                                self.worker_stats[worker_id].model = model_info.get_info(rank)
                            self.model_lock.release()
                            return node_id, worker_list, True
        # 3. Try to get workers on the same node and minimize impact on prewarmed model
        if self.disable_placement:
            for node_id in range(self.num_servers):
                worker_ids = []
                for worker_id in self.node_workers[node_id]:
                    if self.worker_stats[worker_id].model is None:
                        worker_ids.append(worker_id)
                        if len(worker_ids) == world_size:
                            self.model_lock.release()
                            return node_id, tuple(worker_ids), False
        else:
            for node_id in range(self.num_servers):
                worker_ids = []
                for worker_id in self.node_workers[node_id]:
                    if self.worker_stats[worker_id].model is None:
                        worker_ids.append(worker_id)
                if len(worker_ids) >= world_size:
                    if not self.use_unified_memory or len(worker_ids) == world_size:
                        choosed_workers = worker_ids[:world_size]
                    else:
                        # Choose workers with minimal interference
                        worker_num_overlapped_models = [0 for _ in range(len(worker_ids))]
                        for model, prewarm_worker_ids in self.node_models[node_id]:
                            for worker_id in prewarm_worker_ids:
                                if worker_id in worker_ids:
                                    worker_num_overlapped_models[worker_ids.index(worker_id)] += 1
                        worker_scores = [(score, worker_ids[index]) for index, score in enumerate(worker_num_overlapped_models)]
                        worker_scores = sorted(worker_scores, key = lambda x: x[0])
                        choosed_workers = sorted([worker_score[1] for worker_score in worker_scores[:world_size]])
                    for rank, worker_id in enumerate(choosed_workers):
                        self.worker_stats[worker_id].model = model_info.get_info(rank)
                    self.model_lock.release()
                    return node_id, tuple(choosed_workers), False
        
        self.model_lock.release()
        # print(f"_get_available_workers: Not enough workers in the cluster.")
        return None, None, False
    
    """
    Reset all workers.
    """
    def reset_workers(self):
        # TODO: this function does not delete prewarm-related information
        # Reset worker GPUs
        total_gpu_memory = 0
        for worker_id in range(self.num_workers):
            if self.worker_stats[worker_id].model is not None:
                total_gpu_memory += ray.get(self.workers[worker_id].reset_worker.remote(True))
        
        # Reset worker status
        self.worker_stats = []
        for (node_id, gpu_ids) in self.worker_node_gpu_ids:
            self.worker_stats.append(WorkerStatus(node=node_id, gpu_id=gpu_ids[0]))
        self.gpu_mem = [self.mem_per_gpu for _ in range(self.num_servers)]

        # Reset engines by re-initialization
        procs_to_kill = []
        for id, proc in self.running_engines.items():
            procs_to_kill.append(proc)
        for id, proc in self.prewarmed_engines:
            procs_to_kill.append(proc)
        for proc in procs_to_kill:
            proc.kill()
        for proc in procs_to_kill:
            proc.wait()
        self.prewarmed_engines = []
        self.running_engines = {}
        self.engine_stats = {}
        if self.enable_kv_prewarm:
            for data in self.engine_data:
                data.destroy()
            gc.collect()
            self.engine_data = [EngineData(i) for i in range(self.max_concurrent_engines)]
        if not self.disable_all_prewarm:
            self.prewarmed_engines = self._init_engines(self.max_concurrent_engines)

        return total_gpu_memory
    
    def _initialize_ray_cluster(self):
        device_str = "GPU"
        current_placement_group = ray.util.get_current_placement_group()
        if current_placement_group:
            # We are in a placement group
            bundles = current_placement_group.bundle_specs
            # Verify that we can use the placement group
            world_size = 0
            for bundle in bundles:
                bundle_devices = bundle.get(device_str, 0)
                if bundle_devices > 1:
                    raise ValueError(f"Placement group bundle cannot have more than 1 {device_str}.")
                if bundle_devices:
                    world_size += 1
        else:
            world_size = int(ray.cluster_resources().get(device_str, 0))
            placement_group_specs: List[Dict[str, float]] = ([{
                device_str: 1.0
            } for _ in range(world_size)])
            # By default, Ray packs resources as much as possible.
            current_placement_group = ray.util.placement_group(placement_group_specs, strategy="PACK", name="prewarming_group")
            _wait_until_pg_ready(current_placement_group)
        return current_placement_group

    def _init_workers_ray(self, placement_group, use_loaded_model: bool):
        worker_wrapper_kwargs = self._get_worker_wrapper_args()

        workers = []
        node_workers = []       # node id -> list of worker ranks
        worker_node_gpu_ids = []

        node_worker_dict = defaultdict(list)

        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            worker = ray.remote(
                num_cpus=0,
                num_gpus=1,
                runtime_env={"env_vars": self.vllm_env},
                scheduling_strategy=scheduling_strategy,
            )(MyWorkerWrapper).remote(**worker_wrapper_kwargs)
            node_gpu_id = ray.get(worker.get_node_and_gpu_ids.remote())
            node_id = node_gpu_id[0]
            gpu_id = node_gpu_id[1][0]
            if gpu_id in self.disabled_gpus:
                continue
            node_worker_dict[node_id].append((node_gpu_id[1], worker))

        node_list = list(node_worker_dict.keys())
        for index, node_id in enumerate(node_list):
            worker_list = []
            for gpu_ids, worker in node_worker_dict[node_id]:
                worker_list.append(len(workers))
                workers.append(worker)
                worker_node_gpu_ids.append((index, gpu_ids))
            node_workers.append(worker_list)
        
        stime = time.time()
        prewarm_device_futures = []
        init_prewarm_loader_futures = []
        for i, (node_id, gpu_ids) in enumerate(worker_node_gpu_ids):
            prewarm_device_futures.append(workers[i].prewarm_device.remote(gpu_ids[0]))
            init_prewarm_loader_futures.append(workers[i].init_prewarm_loader.remote(self.load_config))
        
        worker_gpu_mem = ray.get(prewarm_device_futures)
        ray.get(init_prewarm_loader_futures)
        etime = time.time()
        print(f"Prewarm device time cost = {'%.2f' % (etime - stime)} seconds")

        worker_gpu_mem = [0 for _ in range(len(workers))]

        if use_loaded_model:
            stime = time.time()
            load_model_futures = []
            for i, (node_id, gpu_ids) in enumerate(worker_node_gpu_ids):
                load_model_futures.append(workers[i].init_model_loader.remote(ModelList))
            ray.get(load_model_futures)
            etime = time.time()
            print(f"Init model loader time cost = {'%.2f' % (etime - stime)} seconds")

        return workers, node_list, node_workers, worker_node_gpu_ids, worker_gpu_mem
    
    def _get_worker_wrapper_args(self) -> Dict[str, Any]:
        (worker_module_name, worker_class_name,
         worker_class_fn) = self._get_worker_module_and_class()
        
        return dict(
            worker_module_name=worker_module_name,
            worker_class_name=worker_class_name,
            worker_class_fn=worker_class_fn,
            # use_ray_compiled_dag=True,
            trust_remote_code=True,
        )
    
    def _get_worker_module_and_class(self) -> Tuple[str, str, Optional[Callable[[], Type[WorkerBase]]]]:
        worker_class_fn = None
        worker_module_name = "vllm.worker.worker"
        worker_class_name = "Worker"
        return (worker_module_name, worker_class_name, worker_class_fn)
    
    def _init_engines(self, num_engines: int = 1, worker_ids: Optional[Tuple[int]] = None):
        stime = time.time()
        engines = []
        env = self.vllm_env.copy()
        engine_ids = []
        if worker_ids:
            target_bundle_ids = ",".join([str(worker_id) for worker_id in worker_ids])
        else:
            target_bundle_ids = None
        for _ in range(num_engines):
            engine_id = self.prewarm_engine_counter
            engine_ids.append(engine_id)
            self.prewarm_engine_counter += 1
            stdout_file_name = f"engine_stdout{engine_id}.out"
            stderr_file_name = f"engine_stderr{engine_id}.out"
            f_out = open(stdout_file_name, 'w')
            f_err = open(stderr_file_name, 'w')
            # Permit programmer to read the files in user mode
            os.chmod(stdout_file_name, 0o644)
            os.chmod(stderr_file_name, 0o644)

            env["VLLM_PREWARM_ENGINE_ID"] = str(engine_id)
            if target_bundle_ids:
                env["TARGET_BUNDLE_IDS"] = target_bundle_ids
            process = subprocess.Popen(
                [sys.executable, '-m', 'vllm.entrypoints.openai.api_server'],
                env=env,
                stdin=subprocess.PIPE,
                stdout=f_out,
                stderr=f_err,
                text=True,
            )
            engines.append((engine_id, process))

            f_out.close()
            f_err.close()

        if not self.disable_all_prewarm:
            # Wait for engine startup
            for id in engine_ids:
                stdout_file_name = f"engine_stdout{engine_id}.out"
                with open(stdout_file_name, 'r') as f:
                    while True:
                        line = f.readline()
                        if 'READY%' in line:
                            break
        print(f"Prepared {num_engines} engines. Time cost = {'%.2f' % (time.time() - stime)} seconds.")
        return engines

    """
    Start a prewarmed engine
    """
    def start_prewarmed_engine(self, model: str, worker_ids: Tuple[int], args_str: Optional[str] = None):
        if self.disable_all_prewarm:
            # Start a new vLLM engine on demand
            # Pass $worker_ids to let the engine use specific Ray bundles
            prewarm_engine_id, engine = self._init_engines(1, worker_ids)[0]
        else:
            self.engine_lock.acquire()
            if len(self.prewarmed_engines) < self.engine_runout_threshold:
                if not self.prewarmed_engine_extend_future:
                    self.prewarmed_engine_extend_future = self.engine_extend_executor.submit(self._init_engines, self.max_concurrent_engines)
                extended_engines = None
                if not self.prewarmed_engines:
                    print(f"Warning: No available prewarmed engine. Waiting for creating new engines.")
                    stime = time.time()
                    extended_engines = self.prewarmed_engine_extend_future.result()
                    print(f"Waiting for engine creation time cost = {'%.2f' % (time.time() - stime)} seconds")
                elif self.prewarmed_engine_extend_future.done():
                    extended_engines = self.prewarmed_engine_extend_future.result()
                if extended_engines:
                    self.prewarmed_engines.extend(extended_engines)
                    self.prewarmed_engine_extend_future = None
            (prewarm_engine_id, engine) = self.prewarmed_engines.pop()
            self.engine_lock.release()
        self.running_engines[prewarm_engine_id] = engine

        if args_str is None:
            return "Error: engine args is needed."
        
        engine.stdin.write(args_str + "\n")
        engine.stdin.flush()

        stdout_file_name = f"engine_stdout{prewarm_engine_id}.out"
        stderr_file_name = f"engine_stderr{prewarm_engine_id}.out"

        # print(f"Engine Started. Stdout is saved to {stdout_file_name}. Stderr is saved = {stderr_file_name}.")
        return prewarm_engine_id
    
    """
    background_workers: ranks of workers that should load model in background.
    """
    def prewarm_model_async(self, model: str, tp_size: int, pp_size: int, worker_ids: Tuple[int], background_workers: Optional[List[int]] = None):
        if not self.use_unified_memory:
            raise ValueError("Prewarm model needs unified memory.")
        assert tp_size * pp_size == len(worker_ids)
        futures = []
        model_infos = []
        for rank in range(len(worker_ids)):
            pp_rank = rank // tp_size
            tp_rank = rank % tp_size
            model_info = ModelInfo(model, tensor_parallel_size=tp_size, tensor_parallel_rank=tp_rank, pipeline_parallel_size=pp_size, pipeline_parallel_rank=pp_rank)
            model_infos.append(model_info)
        
        # Get world_id
        world_id = str(self.world_counter)
        self.world_counter += 1

        print(f"Prewarming model {model} on {worker_ids}. Background workers = {background_workers}. World = {world_id}.")

        for rank, worker_id in enumerate(worker_ids):
            is_background = False
            if background_workers and rank in background_workers:
                is_background = True
            futures.append(self.workers[worker_id].load_model.remote(model_infos[rank], worker_ids, world_id, is_background))

        model_info = ModelInfo(model, tensor_parallel_size=tp_size, pipeline_parallel_size=pp_size)
        return (model_info, worker_ids, futures, list(range(len(worker_ids))))
    
    def prewarm_model_sync(self, model: str, tp_size: int, pp_size: int, worker_ids: Tuple[int], background_workers: Optional[List[int]] = None):
        model_info, worker_ids, futures, unfin_workers = self.prewarm_model_async(model, tp_size, pp_size, worker_ids, background_workers)
        result = ray.get(futures)
        for rank in unfin_workers:
            while not ray.get(self.workers[worker_ids[rank]].check_prewarm_compl.remote(model_info.get_info(rank), worker_ids)):
                time.sleep(0.1)
        self.model_workers[model].add(worker_ids)
        node_id = worker_ids[0] // self.num_gpu_per_server
        self.node_models[node_id].add((model, worker_ids))
        print(f"Prewarmed model {model_info.model_name} with tp {tp_size} on {worker_ids}")
        return "success"
    
    def _check_prewarming_models(self):
        for node_id in self.in_prewarming_models.keys():
            in_prewarming_model_list_ = []
            for model_info, worker_ids, futures, unfin_workers in self.in_prewarming_models[node_id]:
                remaining_refs = []
                if futures:
                    ready_refs, remaining_refs = ray.wait(futures, timeout=0)
                if remaining_refs:
                    in_prewarming_model_list_.append((model_info, worker_ids, remaining_refs, unfin_workers))
                else:
                    # All futures complete
                    futures = [self.workers[worker_ids[rank]].check_prewarm_compl.remote(model_info.get_info(rank), worker_ids) for rank in unfin_workers]
                    try:
                        results = ray.get(futures, timeout=1)
                    except Exception as e:
                        print(f"Check in_prewarming_model meet exception: {e}")
                        in_prewarming_model_list_.append((model_info, worker_ids, remaining_refs, unfin_workers))
                        continue
                    if not all(results):
                        # Still a worker not completed.
                        unfin_workers_ = []
                        for index, result in enumerate(results):
                            if not result:
                                unfin_workers_.append(unfin_workers[index])
                        in_prewarming_model_list_.append((model_info, worker_ids, remaining_refs, unfin_workers_))
                        continue
                    self.model_workers[model_info.model_name].add(worker_ids)
                    self.node_models[node_id].add((model_info.model_name, worker_ids))
                    print(f"Prewarmed model {model_info.model_name} with tp {model_info.tensor_parallel_size} on {worker_ids}")
            self.in_prewarming_models[node_id] = in_prewarming_model_list_
    
    def init_timer(self):
        if self.use_unified_memory and self.enable_prewarm:
            cur_time = time.time()
            for model, status in self.model_stats.items():
                status.last_time = cur_time
            self.prewarm_scheduler.set_time_start(cur_time)

    def prewarm_model_loop(self):
        while True:
            self.model_lock.acquire()
            self._check_prewarming_models()
            self.model_lock.release()

            if self.enable_prewarm:
                # Update current loads
                self.engine_lock.acquire()
                self.prewarm_scheduler.update_loads(self.model_stats)
                self.engine_lock.release()

                # Add in-prewarming models
                self.model_lock.acquire()
                node_models = copy.deepcopy(self.node_models)
                for node_id, in_prewarming_model_list in self.in_prewarming_models.items():
                    for model_info, worker_ids, futures, unfin_workers in in_prewarming_model_list:
                        node_models[node_id].add((model_info.model_name, worker_ids))
                prewarm_model_list = self.prewarm_scheduler.prewarm(self.gpu_mem, node_models, self.num_engines)    # Note that gpu_mem may be changed during scheduling
                if prewarm_model_list:
                    print(f"Plan to prewarm {len(prewarm_model_list)} new models.")
                    new_prewarming_models = []
                    for model, worker_ids in prewarm_model_list:
                        node_id = worker_ids[0] // self.num_gpu_per_server
                        background_workers = []
                        for worker_id in worker_ids:
                            if self.worker_stats[worker_id].model:
                                # Note that these workers may become idle before we call prewarm_model_async, but this will not affect correctness since prewarming model in a thread does not affect speed.
                                background_workers.append(worker_id)
                        ret = self.prewarm_model_async(model, self.model_info[model][1], 1, worker_ids, background_workers)
                        new_prewarming_models.append((node_id, ret))
                    for node_id, ret in new_prewarming_models:
                        self.in_prewarming_models[node_id].append(ret)
                self.model_lock.release()
            time.sleep(0.5)
    
    def _check_engines(self):
        # 1. Autoscaling model engines
        self.engine_lock.acquire()
        engines_to_create = []
        cur_time = time.time()
        print(f"-----AutoScaler-----")
        for model, status in self.model_stats.items():
            num_engines = self.num_engines[model] + self.num_in_creating_engines[model]
            mx_load = num_engines * BATCH_SIZE
            cur_num_reqs = status.get_avg_load(cur_time)
            print(f"Model {model}: mx_load = {mx_load}, cur_num_reqs = {cur_num_reqs}")
            if cur_num_reqs < mx_load * SCALE_LOWER_BOUND:
                num_engines_to_stop = num_engines - math.ceil(cur_num_reqs / SCALE_LOWER_BOUND / BATCH_SIZE)
                if num_engines_to_stop > 0:
                    engines_can_stop = []
                    for engine_id in self.model_engines[model]:
                        engine_stat = self.engine_stats[engine_id]
                        if cur_time - engine_stat.time_created > PROTECT_PERIOD:
                            engines_can_stop.append(engine_stat)
                    if not engines_can_stop:
                        continue
                    print(f"Plan to stop {len(engines_can_stop)} engines for {model}. Avg Load = {'%.2f' % cur_num_reqs}. Num Engines = {self.num_engines[model]}. Num in-Creating Engines = {self.num_in_creating_engines[model]}.")
                    num_engines_to_stop = min(num_engines_to_stop, len(engines_can_stop))
                    sorted_engines = sorted(engines_can_stop, key=lambda x: x.num_reqs)
                    stop_engines = sorted_engines[:num_engines_to_stop]
                    for engine_stat in stop_engines:
                        print(f"Stopping Engine {engine_stat.id}...")
                        engine_stat.stopping = True
                        # Remove from scheduling list
                        self.model_engines[model].remove(engine_stat.id)
                        self.num_engines[model] -= 1
                        status.stopping_engines.add(engine_stat.id)
            elif cur_num_reqs > mx_load * SCALE_UPPER_BOUND:
                num_engines_to_create = math.ceil(cur_num_reqs / SCALE_UPPER_BOUND / BATCH_SIZE) - num_engines
                if num_engines_to_create > 0:
                    print(f"Plan to create {num_engines_to_create} engines for {model}. Avg Load = {'%.2f' % cur_num_reqs}. Num Engines = {self.num_engines[model]}. Num in-Creating Engines = {self.num_in_creating_engines[model]}.")
                    engines_to_create.append((model, num_engines_to_create))
                    self.num_in_creating_engines[model] += num_engines_to_create
        self.engine_lock.release()

        resource_available = True
        for model, num_engines_to_create in engines_to_create:
            if not resource_available:
                self.num_in_creating_engines[model] -= num_engines_to_create
                continue
            for index in range(num_engines_to_create):
                engine_id = self.create_engine(ModelInfo(model, tensor_parallel_size=self.model_info[model][1]), False)
                if engine_id is None:
                    resource_available = False
                    self.engine_lock.acquire()
                    self.num_in_creating_engines[model] -= num_engines_to_create - index
                    break
        if not resource_available:
            self.engine_lock.release()
        # 2. Check running engines
        self.engine_lock.acquire()
        engine_stats_ = {}
        for idx, engine_stat in self.engine_stats.items():
            if engine_stat.stopping:
                if engine_stat.num_reqs == 0:
                    # Stop this engine
                    stime = time.time()
                    if engine_stat.prewarm_engine_id != -1:
                        self.running_engines[engine_stat.prewarm_engine_id].kill()
                        del self.running_engines[engine_stat.prewarm_engine_id]
                    futures = []
                    self.model_stats[engine_stat.model].stopping_engines.remove(idx)
                    self.model_lock.acquire()
                    if not self.disable_all_prewarm:
                        for worker_id in engine_stat.workers:
                            futures.append(self.workers[worker_id].reset_worker.remote(False if self.use_unified_memory else True))
                    for worker_id in engine_stat.workers:
                        self.worker_stats[worker_id].model = None
                    if not self.disable_all_prewarm:
                        ret = ray.get(futures)
                        for index, worker_id in enumerate(engine_stat.workers):
                            self.gpu_mem[worker_id] = ret[index]
                    self.model_lock.release()
                    if self.enable_kv_prewarm:
                        self.engine_data[idx].destroy()
                    print(f"Engine {idx} stopped. Time cost = {'%.1f' % (time.time() - stime)} second")
                else:
                    engine_stats_[idx] = engine_stat
                    if self.enable_kv_prewarm:
                        # Inform allocator to release extra blocks 
                        engine_data = self.engine_data[idx]
                        if engine_data.total_blocks is None:
                            print(f"Error: Engine {idx} does not register KV. Skip.")
                        else:
                            target_free_blocks = int(engine_data.total_blocks * engine_stat.num_reqs / BATCH_SIZE)
                            engine_data.kv_shm.buf[:4] = struct.pack("i", target_free_blocks)
                            cur_reserved_blocks = struct.unpack("i", engine_data.kv_shm.buf[4:8])[0]
                            if engine_data.last_reserved_blocks < cur_reserved_blocks:
                                # New block reserved
                                block_ids_to_reserve = np.frombuffer(engine_data.kv_shm.buf[8+engine_data.last_reserved_blocks*4:8+cur_reserved_blocks*4], dtype=np.int32)
                                for worker_id in engine_stat.workers:
                                    self.workers[worker_id].release_blocks.remote(block_ids_to_reserve, engine_data.block_size, engine_data.total_blocks)
                                self.model_lock.acquire()
                                for worker_id in engine_stat.workers:
                                    self.gpu_mem[worker_id] += len(block_ids_to_reserve) * engine_data.block_size
                                self.model_lock.release()
                                engine_data.last_reserved_blocks = cur_reserved_blocks
            else:
                engine_stats_[idx] = engine_stat
        self.engine_stats = engine_stats_
        self.engine_lock.release()

    def check_engine_loop(self):
        while True:
            self._check_engines()
            time.sleep(0.5)
    
    def print_loop(self):
        while True:
            self.engine_lock.acquire()
            print(f"-----Engine Status-----")
            for idx, engine_stat in self.engine_stats.items():
                print(f"Engine {idx} ({engine_stat.model}): num_reqs = {engine_stat.num_reqs}, stopping = {engine_stat.stopping}.")
            self.engine_lock.release()
            self.model_lock.acquire()
            print(f"-----Worker Status-----")
            free_workers = []
            for worker_id in range(self.num_workers):
                if self.worker_stats[worker_id].model is None:
                    free_workers.append(worker_id)
            print(f"Free Workers = {free_workers}")
            self.model_lock.release()
            time.sleep(5)

    """
    --- Scheduling Functions ---
    """
    def _update_model_req(self, model: str, delta: int):
        status = self.model_stats[model]
        cur_time = time.time()
        if self.use_unified_memory and self.enable_prewarm:
            status.sum_loads += (cur_time - status.last_time) * status.num_reqs
            status.num_reqs += delta
            status.last_time = cur_time
            if delta > 0:
                status.max_loads = max(status.max_loads, status.num_reqs)
        else:
            # Just update the num_reqs field
            status.num_reqs += delta
        status.load_stamp.append((cur_time, status.num_reqs))

    def stop_engine(self, engine_id: int):
        self.engine_lock.acquire()
        self.engine_stats[engine_id].stopping = True
        self.engine_lock.release()
    
    def compl_request(self, engine_id: int):
        if self.is_muxserve:
            self.engine_reqs[engine_id] -= 1
            return
        self.engine_lock.acquire()
        self._update_model_req(self.engine_stats[engine_id].model, -1)
        self.engine_stats[engine_id].num_reqs -= 1
        self.engine_lock.release()

    # CORE scheduler: scheduling workers for the model and return endpoint address. If no available workers, return None.
    def create_engine(self, model_info: ModelInfo, direct_allocation: bool) -> str:
        node_id, worker_ids, prewarm_hit = self._get_available_workers(model_info, self.node_id, model_info.world_size)
        if not worker_ids:
            return None
        
        # Mark these workers as running
        pp_rank = 0
        tp_rank = 0
        self.model_lock.acquire()
        if not self.disable_all_prewarm and not self.use_unified_memory:
            distributed_init_method = get_distributed_init_method("127.0.0.1", get_open_port())
            world_id = str(self.world_counter)
            self.world_counter += 1
        for worker_id in worker_ids:
            model_info_ = ModelInfo(model_info.model_name, tp_rank, model_info.tensor_parallel_size, pp_rank, model_info.pipeline_parallel_size)
            self.worker_stats[worker_id].model = model_info_
            self.gpu_mem[worker_id] = 0

            if not self.disable_all_prewarm:
                # Update model information in worker wrapper
                if self.use_unified_memory:
                    # Existing prewarming process are stopped and prewarmed models are evicted
                    self.workers[worker_id].update_model_info.remote(model_info_, worker_ids)
                else:
                    # No prewarming. We need to configure the same distributed_init_method for workers
                    self.workers[worker_id].update_model_info.remote(model_info_, worker_ids, distributed_init_method, world_id)

            if tp_rank == model_info.tensor_parallel_size - 1:
                pp_rank += 1
                tp_rank = 0
            else:
                tp_rank += 1

        if self.use_unified_memory:
            # Stop existing prewarming process
            _in_prewarming_models = []
            worker_ids_set = set(worker_ids)
            for prewarm_model_info, prewarm_worker_ids, futures, unfin_workers in self.in_prewarming_models[node_id]:
                prewarm_worker_ids_set = set(prewarm_worker_ids)
                if worker_ids_set & prewarm_worker_ids_set:
                    if prewarm_model_info != model_info or worker_ids != prewarm_worker_ids:
                        print(f"Stop prewarming model {prewarm_model_info.model_name} on {prewarm_worker_ids}...")
                        continue
                _in_prewarming_models.append((prewarm_model_info, prewarm_worker_ids, futures, unfin_workers))
            self.in_prewarming_models[node_id] = _in_prewarming_models
            
            # Remove other prewarmed models on these workers
            conflict_models = set()
            for model, prewarm_worker_ids in self.node_models[node_id]:
                prewarm_worker_ids_set = set(prewarm_worker_ids)
                if worker_ids_set & prewarm_worker_ids_set:
                    if model != model_info.model_name or worker_ids != prewarm_worker_ids:
                        print(f"Remove prewarmed model {model} on {prewarm_worker_ids}.")
                        conflict_models.add((model, prewarm_worker_ids))
            if conflict_models:
                self.node_models[node_id] -= conflict_models
                for model, prewarm_worker_ids in conflict_models:
                    self.model_workers[model].remove(prewarm_worker_ids)

            # Prewarm the model
            if not prewarm_hit:
                ret = self.prewarm_model_async(model_info.model_name, model_info.tensor_parallel_size, model_info.pipeline_parallel_size, worker_ids)
                self.in_prewarming_models[node_id].append(ret)
        self.model_lock.release()
        
        # Start a prewarmed engine
        args = f"--model {model_info.model_name} --max-num-seqs=256 --max-model-len=2048 --max-num-batched-tokens=100000 --tensor-parallel-size={model_info.tensor_parallel_size} --pipeline-parallel-size=1 --dtype=float16 --enforce-eager --block-size=256 --host=0.0.0.0 --port={5050+self.engine_counter} --gpu-memory-utilization=0.9 --trust-remote-code --distributed-executor-backend=ray --disable-frontend-multiprocessing --load-format=sharded_state"
        prewarm_engine_id = self.start_prewarmed_engine(model_info.model_name, worker_ids, args)
        
        # Register a new engine
        self.engine_lock.acquire()
        engine_id = self.engine_counter
        self.engine_counter += 1
        self.engine_stats[engine_id] = EngineStatus(id=engine_id, model=model_info.model_name, workers=worker_ids, prewarm_engine_id=prewarm_engine_id, time_created=time.time())
        if direct_allocation:
            self.engine_stats[engine_id].num_reqs += 1
        self.model_engines[model_info.model_name].add(engine_id)
        self.num_engines[model_info.model_name] += 1
        self.num_in_creating_engines[model_info.model_name] -= 1

        # Check whether we have enough engine_data prepared
        if self.enable_kv_prewarm:
            if len(self.engine_data) - engine_id < self.engine_runout_threshold:
                if not self.engine_data_extend_future:
                    def extend_engine_data(start_idx: int, num_engine: int):
                        engine_data = []
                        for idx in range(start_idx, start_idx + num_engine):
                            engine_data.append(EngineData(idx))
                        return engine_data
                    self.engine_data_extend_future = self.engine_extend_executor.submit(extend_engine_data, len(self.engine_data), self.max_concurrent_engines)
                extended_engine_data = None
                if len(self.engine_data) <= engine_id:
                    print(f"Warning: No available engine data. Waiting for creating new data.")
                    stime = time.time()
                    extended_engine_data = self.engine_data_extend_future.result()
                    print(f"Waiting for data creation time cost = {'%.2f' % (time.time() - stime)} seconds")
                elif self.engine_data_extend_future.done():
                    extended_engine_data = self.engine_data_extend_future.result()
                if extended_engine_data:
                    self.engine_data.extend(extended_engine_data)
                    self.engine_data_extend_future = None
        self.engine_lock.release()
        
        self.prewarm_engine_info[prewarm_engine_id] = (worker_ids, engine_id)

        print(f"Engine {engine_id} for {model_info.model_name} started on {worker_ids} [Hit = {prewarm_hit}] with prewarm engine id {prewarm_engine_id}.")

        return engine_id
    
    def schedule(self, request_id: int, model: str):
        if model not in self.model_info:
            return -1
        
        if self.is_muxserve:
            if model not in self.model_engines:
                raise ValueError(f"Model {model} not found in model_engines")
            for engine_id in self.model_engines[model]:
                if self.engine_reqs[engine_id] < self.batch_size[engine_id]:
                    self.engine_reqs[engine_id] += 1
                    return engine_id
            return None
        
        self.engine_lock.acquire()
        if request_id not in self.recorded_reqs:
            self._update_model_req(model, 1)
            self.recorded_reqs.add(request_id)
        if model in self.model_engines:
            mn_reqs = BATCH_SIZE
            mn_reqs_engine_id = None
            for engine_id in self.model_engines[model]:
                if self.engine_stats[engine_id].num_reqs < mn_reqs:
                    mn_reqs = self.engine_stats[engine_id].num_reqs
                    mn_reqs_engine_id = engine_id
            if mn_reqs_engine_id is not None:
                self.engine_stats[mn_reqs_engine_id].num_reqs += 1
                self.engine_lock.release()
                return mn_reqs_engine_id
        if self.num_in_creating_engines[model] > 0:
            # Engine is in creating, return None currently
            self.engine_lock.release()
            return None
        self.num_in_creating_engines[model] += 1        # Set in_creating_engines
        self.engine_lock.release()
        
        # Create a new engine
        engine_id = self.create_engine(ModelInfo(model, tensor_parallel_size=self.model_info[model][1]), True)

        if engine_id is None:
            # No new engine is created
            self.engine_lock.acquire()
            self.num_in_creating_engines[model] -= 1
            self.engine_lock.release()
        return engine_id
        
prewarm_manager = None

"""
Request Handling
"""
clients = {}
class RequestModel(BaseModel):
    id: int
    model: str
    prompt: str
    max_tokens: Optional[int] = 100
    stream: Optional[bool] = False

app = fastapi.FastAPI()

Timer_inited = False

@app.post('/')
async def request_handler(request: RequestModel, raw_request: Request, background_task: BackgroundTasks):
    global prewarm_manager
    global clients
    global Timer_inited

    stime = time.time()
    request_id = request.id
    prompt = request.prompt
    max_tokens = request.max_tokens
    model = request.model
    stream = request.stream
    print(f"Received request [{request_id}] for model {model}")

    if not Timer_inited:
        # This is the first request. Init timer.
        prewarm_manager.init_timer.remote()
        Timer_inited = True

    engine_id = await prewarm_manager.schedule.remote(request_id, model)

    if engine_id == -1:
        return JSONResponse(content="Model not found in ModelList",
                            status_code=404)

    count = 0
    while engine_id is None:
        count += 1
        if count == 60:
            print(f"WARNING: Request [{request_id}] for Model {model} cannot find an engine for 60 seconds")
            count = 0
        await asyncio.sleep(1)
        engine_id = await prewarm_manager.schedule.remote(request_id, model)
    engine_addr = "http://localhost:" + str(5050+engine_id) + "/v1"

    # Wait for server startup
    while True:
        try:
            response = requests.get(engine_addr, timeout=1)
            break
        except requests.exceptions.RequestException as e:
            await asyncio.sleep(0.05)

    etime = time.time()
    print(f"Request [{request_id}] allocated to Engine {engine_id}. Time cost = {'%.1f' % (etime - stime)} seconds.")

    prompt = [{"role": "user", "content": prompt}]
    if stream:
        if engine_addr not in clients:
            clients[engine_addr] = AsyncOpenAI(
                base_url=engine_addr,
                api_key="token"
            )
        chat = await clients[engine_addr].chat.completions.create(
            model=model,
            messages=prompt,
            max_tokens=max_tokens,
            extra_body={"ignore_eos": True},
            stream=True,
            stream_options={"include_usage": True},
        )
        async def resp_generator(chat):
            try:
                async for stream_response in chat:
                    if stream_response.choices and stream_response.choices[0].delta.content is not None:
                        yield stream_response.choices[0].delta.content
                if stream_response.usage is None:
                    print(f"Error: stream_response final return has null usage")
                else:
                    # produce num_completion_tokens
                    yield f"#{stream_response.usage.completion_tokens}"
            except Exception as e:
                print(f"Request [{request_id}] meets exception: {e}")
                exc_info = sys.exc_info()
                print("".join(traceback.format_exception(*exc_info)))
            prewarm_manager.compl_request.remote(engine_id)
            print(f"Request [{request_id}] completed. Elapsed = {'%.1f' % (time.time() - etime)} seconds")
        return StreamingResponse(content=resp_generator(chat),
                                 media_type="text/event-stream")
    else:
        if engine_addr not in clients:
            clients[engine_addr] = AsyncOpenAI(
                base_url=engine_addr,
                api_key="token"
            )
        chat = await clients[engine_addr].chat.completions.create(
            model=model,
            messages=prompt,
            max_tokens=max_tokens,
            extra_body={"ignore_eos": True},
            stream=False
        )
        prewarm_manager.compl_request.remote(engine_id)
        print(f"Request [{request_id}] completed. Elapsed = {'%.1f' % (time.time() - etime)} seconds")
        return JSONResponse(content=chat.choices[0].message.content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-format', type=str, default='auto')
    parser.add_argument('--download-dir', type=nullable_str, default=None)
    parser.add_argument('--model-loader-extra-config', type=nullable_str, default=None)
    parser.add_argument('--ignore-patterns', action="append", type=str, default=[])
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--use-unified-memory', action='store_true', help='Allocate a unified memory region for model prewarming. If ture, prewarming is enabled.')
    parser.add_argument('--disabled-gpus', type=str, default='')
    parser.add_argument('--character-file', type=str, default='')
    parser.add_argument('--prewarm-model', action='store_true', help='Prewarm a model in intialization. Need to enable unified memory.')
    parser.add_argument('--prewarm-model-name', type=str, default='')
    parser.add_argument('--prewarm-model-tp-size', type=int, default=1)
    parser.add_argument('--prewarm-model-pp-size', type=int, default=1)
    parser.add_argument('--use-loaded-model', action='store_true', help='Use pre-loaded model.')
    parser.add_argument('--disable-all-prewarm', action='store_true', help='Disable any prewarming.')
    parser.add_argument('--disable-prewarm', action='store_true', help='Disable prewarming new models.')
    parser.add_argument('--disable-kv-prewarm', action='store_true', help='Disable using KV cache for prewarming.')
    parser.add_argument('--disable-placement', action='store_true', help='Disable smart placement when prewarming.')
    parser.add_argument('--muxserve', action='store_true', help='Run MuxServe.')
    parser.add_argument('--placement-config', type=str, default='')
    parser.add_argument('--window-size', type=int, default=300)

    args = parser.parse_args()

    ray.init(ignore_reinit_error=True, namespace="prewarm")
    current_node_id = ray.get_runtime_context().get_node_id()
    prewarm_manager = PrewarmManager.options(name="prewarm_manager", namespace="prewarm", lifetime="detached", scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=current_node_id, soft=False)).remote(args)
    num_workers = ray.get(prewarm_manager.get_num_workers.remote())

    print(f"Num Workers: {num_workers}")
    print(f"Prewarm Manager Initialized.")
    print(f"Start Serving...")

    uvicorn.run(app,
            host='0.0.0.0',
            port=9999,
            log_level='debug',
            timeout_keep_alive=5)
