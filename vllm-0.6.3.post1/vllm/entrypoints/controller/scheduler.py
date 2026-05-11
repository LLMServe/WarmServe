import os
import pickle
import time
import math
import threading
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

from .common import BATCH_SIZE, PCIE_BANDWIDTH, align_up, ModelStatus

@dataclass
class Replica:
    model_id: int = -1
    tp: int = -1
    mem_per_gpu: int = -1
    score: int = -1
    worker_ids: Tuple[int] = ()

"""
Prewarm Scheduler
"""
class Scheduler:
    def __init__(self, model_info: Dict[str, Tuple[int, int, str]], num_servers: int, num_gpu_per_server: int, disable_placement: bool,  window_size: int = 300):
        self.model_info = model_info        # Dict[model -> (size_in_gb, tp, datasource)]
        self.num_servers = num_servers
        self.num_gpu_per_server = num_gpu_per_server
        self.disable_placement = disable_placement
        self.window_size = window_size
        # initialize weights
        self.num_sample_point = 10
        self.weights = [1/(2**i) for i in range(self.num_sample_point)]
        self.sum_weights = [self.weights[0]]
        for i in range(1, self.num_sample_point):
            self.sum_weights.append(self.sum_weights[-1] + self.weights[i])
    
    def init(self, characters: Dict[Tuple[str, str], Tuple[List, List]]):
        self.past_avgs = {}
        self.past_peaks = {}
        self.predicted_avg_load = {}
        self.predicted_peak_load = {}
        self.recorded_avg_load = defaultdict(list)
        self.recorded_peak_load = defaultdict(list)
        self.cur_window_id = 0
        # Read the data of previous days from characters.
        print("-----Prediction Status-----")
        for model_name in self.model_info.keys():
            if model_name not in characters:
                raise ValueError(f"Model {model_name} not found in characteristics file.")
            self.past_avgs[model_name], self.past_peaks[model_name] = characters[model_name]
            self.predicted_avg_load[model_name] = self.past_avgs[model_name][0]
            self.predicted_peak_load[model_name] = self.past_peaks[model_name][0]
            print(f"Model {model_name}: avg_load = {self.predicted_avg_load[model_name]}, peak_load = {self.predicted_peak_load[model_name]}")

    def set_time_start(self, cur_time):
        self.start_time = cur_time
    
    def update_loads(self, model_stats: Dict[str, ModelStatus]) -> bool:
        cur_time = time.time()
        if cur_time - self.start_time < (self.cur_window_id + 1) * self.window_size:
            # It is not the time for next window
            return False
        print("-----Prediction Status-----")
        for model in self.model_info.keys():
            status = model_stats[model]
            recorded_avg_load = self.recorded_avg_load[model]
            recorded_peak_load = self.recorded_peak_load[model]
            past_avgs = self.past_avgs[model]
            past_peaks = self.past_peaks[model]
            # Update recorded_peak_load in this window
            recorded_avg_load.append((status.sum_loads + (cur_time - status.last_time) * status.num_reqs) / self.window_size)
            recorded_peak_load.append(status.max_loads)
            status.last_time = cur_time
            status.sum_loads = 0
            status.max_loads = 0
            # Predict the peak load in next window
            num_sample_point = min(self.num_sample_point, self.cur_window_id + 1)
            sum_value_peak = 0
            sum_value_avg = 0
            for delta in range(num_sample_point):
                sum_value_avg += (recorded_avg_load[self.cur_window_id-delta] - past_avgs[self.cur_window_id-delta]) * self.weights[delta]
                sum_value_peak += (recorded_peak_load[self.cur_window_id-delta] - past_peaks[self.cur_window_id-delta]) * self.weights[delta]
            self.predicted_avg_load[model] = past_avgs[self.cur_window_id + 1] + sum_value_avg / self.sum_weights[num_sample_point-1]
            self.predicted_peak_load[model] = past_peaks[self.cur_window_id + 1] + sum_value_peak / self.sum_weights[num_sample_point-1]
            print(f"Model {model}: avg_load = {self.predicted_avg_load[model]}, peak_load = {self.predicted_peak_load[model]}")
        self.cur_window_id += 1
        return True
    
    """
    Calculate the scores of replicas for each model.
    @Return:
    n_foundation: number of foundation replicas for each model
    n_burst: number of burst replicas for each model
    scores: score list of all replicas for each model
    """
    def _calc_model_scores(self, model_list: List[str], num_instances: Dict[str, int]):
        n_foundation_list = []
        n_burst_list = []
        scores_list = []
        for model in model_list:
            avg_load = self.predicted_avg_load[model]
            peak_load = self.predicted_peak_load[model]
            cur_instance = num_instances[model]
            coldstart_time = self.model_info[model][0] / self.model_info[model][1] / (2**30) / PCIE_BANDWIDTH
            bursty_factor = peak_load / avg_load

            n_foundation = max(math.ceil(avg_load / BATCH_SIZE) - cur_instance, 0)
            n_burst = max(math.ceil(peak_load / BATCH_SIZE) - cur_instance - n_foundation, 0)
            score_list = []
            for i in range(n_foundation):
                score = math.exp(-i/(n_foundation+n_burst)) * coldstart_time
                score_list.append(score)
            for i in range(n_burst):
                score = math.exp(-(n_foundation+i)/(n_foundation+n_burst)) * coldstart_time * bursty_factor
                score_list.append(score)
            
            n_foundation_list.append(n_foundation)
            n_burst_list.append(n_burst)
            scores_list.append(score_list)
        return n_foundation_list, n_burst_list, scores_list

    """
    Schedule a list of prewarming actions according to current cluster status
    @Params:
    gpu_mem: the free memory of all gpus
    node_models: the prewarmed and in-prewarming model on each server
    num_instances: number of running instances of each model
    """
    def prewarm(self, gpu_mem: List[int], node_models: List[Set[Tuple[str, Tuple[int]]]], num_instances: Dict[str, int]) -> List[Tuple[str, Tuple[int]]]:
        model_list = list(self.model_info.keys())
        num_models = len(model_list)
        model_to_idx = {}
        for i, model in enumerate(model_list):
            model_to_idx[model] = i
        
        n_foundation, n_burst, scores = self._calc_model_scores(model_list, num_instances)

        cur_replica_id = [0 for _ in range(num_models)]

        # Convert $node_models to initial replicas
        orig_replicas = []      # the original replica list
        for node_id in range(self.num_servers):
            for model, worker_ids in node_models[node_id]:
                model_id = model_to_idx[model]
                replica_id = cur_replica_id[model_id]
                cur_replica_id[model_id] += 1
                if replica_id < n_foundation[model_id] + n_burst[model_id]:
                    score = scores[model_id][replica_id]
                else:
                    score = 0
                model_info = self.model_info[model]
                orig_replica = Replica(model_id, model_info[1], align_up(model_info[0] // model_info[1]), score, worker_ids)
                orig_replicas.append(orig_replica)
        
        # Replicas on each GPU
        covered_replicas = [[] for _ in range(len(gpu_mem))]

        for replica in orig_replicas:
            for worker_id in replica.worker_ids:
                covered_replicas[worker_id].append(replica)
        
        # Add additional prewarming replicas 
        new_replicas = []
        total_prewarm_replicas = []
        for model_id in range(num_models):
            total_prewarm_replica = n_foundation[model_id] + n_burst[model_id]
            total_prewarm_replicas.append(total_prewarm_replica)
            if cur_replica_id[model_id] < total_prewarm_replica:
                model_info = self.model_info[model_list[model_id]]
                tp = model_info[1]
                size_per_gpu = align_up(model_info[0] // tp)
                for id in range(cur_replica_id[model_id], total_prewarm_replica):
                    new_replica = Replica(model_id, tp, size_per_gpu, scores[model_id][id])
                    new_replicas.append(new_replica)

        # sort by scores
        new_replicas.sort(key=lambda x: x.score, reverse=True)

        prewarming_actions = []
        
        if self.disable_placement:
            # Round robin
            cur_idx = 0
            while True:
                # Try to place model $cur_idx
                if total_prewarm_replicas[cur_idx] <= 0:
                    break
                total_prewarm_replicas[cur_idx] -= 1
                model_info = self.model_info[model_list[cur_idx]]
                tp = model_info[1]
                size_per_gpu = align_up(model_info[0] // tp)
                node_gpus = [[] for _ in range(self.num_servers)]
                for worker_id, mem in enumerate(gpu_mem):
                    if mem >= size_per_gpu:
                        # Check whether there are existing replicas with the same model
                        has_same_model = False
                        for replica in covered_replicas[worker_id]:
                            if replica.model_id == cur_idx:
                                has_same_model = True
                                break
                        if not has_same_model:
                            node_gpus[worker_id // self.num_gpu_per_server].append(worker_id)
                final_worker_ids = None
                for node_id in range(self.num_servers):
                    available_gpus = node_gpus[node_id]
                    if len(available_gpus) < tp:
                        continue
                    final_worker_ids = tuple(available_gpus[:tp])
                    prewarming_actions.append((model_list[cur_idx], final_worker_ids))
                    new_replica = Replica(cur_idx, tp, size_per_gpu, 0)
                    new_replica.worker_ids = final_worker_ids
                    for worker_id in final_worker_ids:
                        covered_replicas[worker_id].append(new_replica)
                        gpu_mem[worker_id] -= new_replica.mem_per_gpu
                    break
                if not final_worker_ids:
                    break
                cur_idx = (cur_idx + 1) % num_models
            return prewarming_actions

        # Try to place these replicas in order
        for new_replica in new_replicas:
            # 1. Find available gpus
            node_gpus = [[] for _ in range(self.num_servers)]
            for worker_id, mem in enumerate(gpu_mem):
                if mem >= new_replica.mem_per_gpu:
                    node_gpus[worker_id // self.num_gpu_per_server].append(worker_id)
            # 2. Find available groups that satisfy our contention requirements
            groups = []
            for node_id in range(self.num_servers):
                available_gpus = node_gpus[node_id]
                num_available_gpus = len(available_gpus)
                tp = new_replica.tp
                if num_available_gpus < tp:
                    continue
                
                """
                gpu_idx: current searching index in $available_gpus
                considered_replicas: replicas that we have considered
                banned_workers: workers that cannot be selected
                cur_workers: current selected workers
                max_score: maximum score of overlapped replicas
                sum_score: sum of scores of overlapped replicas
                groups: total available groups
                """
                def search_for_available_groups(gpu_idx: int, considered_replicas: List[Replica], cur_workers: List[int], banned_workers: List[int], max_score: int, sum_score: int, groups: List[Tuple[int]]):
                    if num_available_gpus - gpu_idx + len(cur_workers) < tp:
                        return
                    if gpu_idx == num_available_gpus:
                        # Transform gpu_id to worker_id
                        workers = tuple([available_gpus[gpu_id] for gpu_id in cur_workers])
                        groups.append((workers, max_score, sum_score))
                        return
                    
                    # Consider select this gpu
                    can_select = True
                    cur_workers_ = cur_workers.copy()
                    if gpu_idx not in cur_workers_:
                        if gpu_idx in banned_workers or len(cur_workers_) == tp:
                            can_select = False
                        else:
                            cur_workers_.append(gpu_idx)
                    if can_select:
                        # Check replicas
                        max_score_ = max_score
                        sum_score_ = sum_score
                        banned_workers_ = banned_workers.copy()
                        for replica in covered_replicas[available_gpus[gpu_idx]]:
                            if replica in considered_replicas:
                                continue
                            if replica.model_id == new_replica.model_id:
                                # Do not prewarm two identical models on the same GPU
                                can_select = False
                                break
                            if replica.tp == tp:
                                # Our GPU set must be the same with the replica
                                for gpu_id in cur_workers_:
                                    if available_gpus[gpu_id] not in replica.worker_ids:
                                        can_select = False
                                        break
                                if not can_select:
                                    break
                                for worker_id in replica.worker_ids:
                                    if worker_id not in available_gpus:
                                        can_select = False
                                        break
                                    id = available_gpus.index(worker_id)
                                    if id not in cur_workers_:
                                        if id < gpu_idx or id in banned_workers_ or len(cur_workers_) == tp:
                                            can_select = False
                                            break
                                        cur_workers_.append(id)
                            elif replica.tp < tp:
                                # Our GPU set must contain this replica
                                for worker_id in replica.worker_ids:
                                    if worker_id not in available_gpus:
                                        can_select = False
                                        break
                                    id = available_gpus.index(worker_id)
                                    if id not in cur_workers_:
                                        if id < gpu_idx or id in banned_workers_ or len(cur_workers_) == tp:
                                            can_select = False
                                            break
                                        cur_workers_.append(id)
                            else:   # replica.tp > tp
                                # Our GPU set must be a subset of this replica. In other words, we should ban all other workers not appear in the replica
                                for gpu_id in range(num_available_gpus):
                                    if available_gpus[gpu_id] not in replica.worker_ids:
                                        # Ban this gpu_id
                                        if gpu_id in banned_workers_:
                                            continue
                                        if gpu_id in cur_workers_:
                                            can_select = False
                                            break
                                        banned_workers_.append(gpu_id)
                            if not can_select:
                                break
                            max_score_ = max(max_score_, replica.score)
                            sum_score_ += replica.score
                    
                    # Select this GPU
                    if can_select:
                        num_added_replica = 0
                        for replica in covered_replicas[available_gpus[gpu_idx]]:
                            if replica not in considered_replicas:
                                considered_replicas.append(replica)
                                num_added_replica += 1
                        search_for_available_groups(gpu_idx + 1, considered_replicas, cur_workers_, banned_workers_, max_score_, sum_score_, groups)
                        if num_added_replica:
                            del considered_replicas[-num_added_replica:]

                    # Do not select this GPU
                    if gpu_idx not in cur_workers:
                        search_for_available_groups(gpu_idx + 1, considered_replicas, cur_workers, banned_workers, max_score, sum_score, groups)

                # Perform search for GPUs on the node
                search_for_available_groups(0, [], [], [], 0, 0, groups)
            if not groups:
                # TODO: consider eviction
                continue
            # 3. Select the best group
            master_worker_ids = None
            master_sum_score = 0
            low_worker_ids = None
            low_sum_score = 0
            for gpu_ids, max_score, sum_score in groups:
                if max_score < new_replica.score:
                    # We can become the highest-priority replica
                    if master_worker_ids is None or master_sum_score > sum_score:
                        master_worker_ids = gpu_ids
                        master_sum_score = sum_score
                else:
                    # We are low-priority replica
                    if low_worker_ids is None or low_sum_score > sum_score:
                        low_worker_ids = gpu_ids
                        low_sum_score = sum_score
            
            final_worker_ids = master_worker_ids if master_worker_ids is not None else low_worker_ids
            
            # 4. Place this replica to $final_gpu_ids
            prewarming_actions.append((model_list[new_replica.model_id], final_worker_ids))
            new_replica.worker_ids = final_worker_ids
            for worker_id in final_worker_ids:
                covered_replicas[worker_id].append(new_replica)
                gpu_mem[worker_id] -= new_replica.mem_per_gpu

        return prewarming_actions
