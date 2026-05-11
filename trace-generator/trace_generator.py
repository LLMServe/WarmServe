import argparse
import csv
import psutil
from datetime import datetime
import os
import gc
import pandas as pd
import numpy as np
import yaml
from typing import List, Tuple
import json
import dataclasses
from typing import Any, Dict, Optional
import random
import pickle

from common import time_dict

INTERVAL = 5 * 60
eps = 1e-6

def apply_burstiness(requests, cv, rng=None):
    """Reshape inter-arrival times to achieve target coefficient of variation.

    Args:
        requests: list of tuples, first element is arrival_time, sorted by time.
        cv: coefficient of variation of inter-arrival times.
            cv=0: perfectly regular (all gaps equal)
            cv=1: exponential/Poisson-like
            cv>1: bursty (heavy-tailed gaps)
            cv<0: keep original trace burstiness (no-op)
        rng: numpy random generator for reproducibility.

    Returns:
        New list of tuples with reshaped arrival times, same order.
    """
    if cv < 0 or len(requests) < 2:
        return requests

    if rng is None:
        rng = np.random.default_rng(42)

    n = len(requests)
    arrivals = np.array([r[0] for r in requests])
    total_duration = arrivals[-1] - arrivals[0]

    if total_duration <= 0:
        return requests

    mean_iat = total_duration / (n - 1)

    if cv < 1e-6:
        # Perfectly regular: all gaps equal
        new_arrivals = np.linspace(arrivals[0], arrivals[-1], n)
    else:
        # Gamma distribution: shape k = 1/cv^2, scale = mean_iat * cv^2
        # This gives mean = mean_iat, CV = cv
        k = 1.0 / (cv * cv)
        theta = mean_iat * cv * cv
        new_iats = rng.gamma(k, theta, size=n - 1)
        # Normalize to preserve total duration
        new_iats = new_iats * (total_duration / new_iats.sum())
        new_arrivals = np.empty(n)
        new_arrivals[0] = arrivals[0]
        np.cumsum(new_iats, out=new_arrivals[1:])
        new_arrivals[1:] += arrivals[0]

    # Rebuild tuples with new arrival times
    result = []
    for i, req in enumerate(requests):
        result.append((new_arrivals[i],) + req[1:])
    return result

LOW_MEM = False

@dataclasses.dataclass
class Request:
    """A single request."""
    model_name: str
    slo: Optional[float]
    idx: int
    time_stamp: Dict  # debug only
    data: Any
    submit_time: float = None  # This will be filled later
    prefill_end_time: float = None  # This will be filled later
    decode_submit_time: float = None  # This will be filled later
    end_time: float = None  # This will be filled later
    is_prefill: bool = True
    output: str = None
    output_idx: int = 0
    output_tokens: Optional[List[int]] = None


class Workload:
    """A sorted list of requests."""

    def __init__(self,
                 arrivals: List[float],
                 requests: List[Request],
                 workload_infos: Optional[Dict[str, Any]] = None):
        assert len(arrivals) == len(requests)

        self.arrivals = np.array(arrivals)
        self.requests = requests
        self.workload_infos = workload_infos

    def __len__(self):
        return len(self.arrivals)


    @classmethod
    def merge(cls, *args):
        if len(args) == 1:
            return args[0]

        number = sum(len(x) for x in args)

        merged_arrivals = np.concatenate(tuple(x.arrivals for x in args))
        merged_requests = sum((x.requests for x in args), [])

        sorted_indices = np.argsort(merged_arrivals)

        arrivals = [None] * number
        requests = [None] * number

        for i, j in enumerate(sorted_indices):
            arrivals[i] = merged_arrivals[j]
            requests[i] = merged_requests[j]
            requests[i].idx = i

        return cls(arrivals, requests)

if not LOW_MEM:
    data_processed = {}
    character_processed = {}

def get_inference_time(model_id: str):
    if model_id in time_dict:
        return time_dict[model_id]
    # Try to remove the suffix '/'
    pos = model_id.rfind('/')
    model_id = model_id[:pos]
    if model_id not in time_dict:
        raise ValueError(f"Model {model_id} not recorded in time_dict!")
    return time_dict[model_id]

def get_type_from_path(csv_path: str):
    if "Conv" in csv_path or "conv" in csv_path:
        return "conv"
    return "code"

def process_csv(csv_path: str, day: int = -1):
    global data_processed
    if not LOW_MEM:
        key = (csv_path, day)
        if key in data_processed:
            return data_processed[key]

    df = None
    if not LOW_MEM and day != -1:
        key = (csv_path, -1)
        if key in data_processed:
            df = data_processed[key].copy()
    gc.collect()

    print(f"Processing csv for {csv_path}, day = {day}")

    if df is None:
        # sort csv in time order
        df = pd.read_csv(csv_path, parse_dates=['TIMESTAMP'], low_memory=True)
        df['time'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
        df.drop(columns=['TIMESTAMP'], inplace=True)
        if get_type_from_path(csv_path) == "conv":
            # monday is from day 1
            time_to_monday = 24 * 3600
        else:
            # monday is from day 3
            time_to_monday = 3 * 24 * 3600
        week_time = 7 * 24 * 3600

        first_row_time = df['time'].iloc[0]
        df['invocation'] = (df['time'] - first_row_time).dt.total_seconds()
        df.dropna(inplace=True)
        df.drop(columns=['time'], inplace=True)
        df['invocation'] = df['invocation'] - time_to_monday
        df.loc[df['invocation'] < 0, 'invocation'] += week_time
        df.sort_values('invocation', inplace=True)

    if day != -1:
        hour_elapse = 3600
        day_elapse = 24 * hour_elapse
        day_stime = day * day_elapse
        df['invocation'] = df['invocation'] - day_stime
        df = df[(df['invocation'] >= 0) & (df['invocation'] <= hour_elapse)]
    
    print(f"Complete processing csv for {csv_path}, day = {day}")

    if not LOW_MEM:
        data_processed[(csv_path, day)] = df
    return df

def get_character_from_csv(csv_path: str, interval: int, prefill_time: float, decode_time: float):
    global character_processed
    if not LOW_MEM:
        key = (csv_path, interval, prefill_time, decode_time)
        if key in character_processed:
            return character_processed[key]
    gc.collect()

    print(f"Obtaining character from {csv_path}, interval = {interval}, prefill_time = {prefill_time}, decode_time = {decode_time}")
    
    df = process_csv(csv_path)

    df['elapse'] = (prefill_time + df['GeneratedTokens'] * decode_time) / 1000.0

    time = []
    delta = []  # number of concurrent request
    
    delta = []
    for index, row in df.iterrows():
        time.append(row['invocation'])
        delta.append((row['invocation'], 1))
        delta.append((row['invocation'] + row['elapse'], -1))
    
    delta.sort()
    cur_time = []
    cur_value = []
    cur_req = 0
    for index in range(len(delta)):
        cur_time.append(delta[index][0])
        cur_req += delta[index][1]
        cur_value.append(cur_req)
    
    # Obtain average and peak
    # 1. Peak
    df = pd.DataFrame({
        'time': cur_time,
        'value': cur_value
    })

    df['interval'] = pd.cut(df['time'], 
                        bins=range(0, int(df['time'].max()) + interval, interval),
                        right=False)

    intervals = df.groupby('interval')
    peaks = intervals['value'].max().tolist()

    if len(peaks) < (7 * 24 * 3600 // interval):
        print(f"Error: trace only has {len(peaks)} intervals.")
        exit(1)

    # 2. Average
    delta_pointer = -1
    num_points = len(cur_time)
    value_mean_list = []
    for interval_id in range(len(peaks)):
        st_time = interval_id * interval
        ed_time = st_time + interval
        sum = 0
        # currenly, delta_time[delta_pointer] < st_time and delta_time[delta_pointer+1] >= st_time
        if delta_pointer >= 0:
            if delta_pointer + 1 >= num_points or cur_time[delta_pointer+1] >= ed_time:
                sum += interval * cur_value[delta_pointer]
            else:
                sum += (cur_time[delta_pointer+1] - st_time) * cur_value[delta_pointer]
                delta_pointer += 1
        else:
            delta_pointer += 1
        # intra-interval points
        while delta_pointer + 1 < num_points and cur_time[delta_pointer+1] < ed_time:
            sum += cur_value[delta_pointer] * (cur_time[delta_pointer+1] - cur_time[delta_pointer])
            delta_pointer += 1
        if cur_time[delta_pointer] >= st_time and cur_time[delta_pointer] < ed_time:
            sum += cur_value[delta_pointer] * (ed_time - cur_time[delta_pointer])
        value_mean_list.append(sum / interval)
    means = value_mean_list

    print(f"Complete obtaining character from {csv_path}, interval = {interval}, prefill_time = {prefill_time}, decode_time = {decode_time}")
    if not LOW_MEM:
        character_processed[key] = (means, peaks)
    return means, peaks

def get_workloads_info_from_yaml(models_yaml: str, alpha: float, is_muxserve:bool) -> List[Tuple[str, float]]:
    with open(models_yaml, "r") as fp:
        model_group = yaml.safe_load(fp)

    models = model_group["models"]

    if is_muxserve:
        model_id = [model["id"] for model in models]
    else:
        model_id = [(model["id"], model["name"]) for model in models]

    dataset_source=[model["dataset_source"] for model in models]
    arr=[(x+1)**(-alpha) for x in range(len(model_id))]  # Power law distribution
    arr_sum = sum(arr)
    arr=[x / arr_sum for x in arr]

    return [(id,alpha_rate, data) for id,alpha_rate, data in zip(model_id, arr,dataset_source)]

def generate_workload(workload_infos: List[Tuple[float, List[int], int, int]], 
                      output_file: str,
                      sampled_requests: List[List[Tuple[float, List[int], int, int]]]
                      ) -> None:
    
    workload_num_requests = [len(reqs) for reqs in sampled_requests]
    models = [model for model,_,_ in workload_infos]
    workloads = []
    for i, model_name in enumerate(models):
        arrivals=[req[0] for req in sampled_requests[i]]
        w=Workload(arrivals,[Request(model_name, None, -1, {}, None) for i in range(len(arrivals))])
        for idx in range(len(w)):
            req = sampled_requests[i][idx]
            w.requests[idx].data = (req[1], req[2], req[3])  # (prompt, inputlen, outputlen)

        workloads.append(w)

    workload = Workload.merge(*workloads)

    workload_json = {
        "info": {
            "rates": workload_infos,
            "num_requests": workload_num_requests,
        },
        "arrivals": workload.arrivals.tolist(),
        "requests": [dataclasses.asdict(r) for r in workload.requests]
    }
    with open(output_file, "w") as f:
        json.dump(workload_json, f)

def get_muxserve_placement(model_yaml:str,
                           alpha: float,
                           output_file: str,
                           request_time:int,
                           request_per_second: float) -> None:
    """
    Get the muxserve placement from the model YAML file.
    """
    num_requests = int(request_time* request_per_second)
    workload_info=get_workloads_info_from_yaml(model_yaml,alpha,True)
    print(f"get workload info: {workload_info}")

    start_date = max(0, 7 - len(workload_info))
    if start_date == 0:
        raise ValueError("Too many Models! The first model cannot get character!")
    sampled_requests = []
    for idx,(model_id,alpha_rate, dataset_source) in enumerate(workload_info):

        df = process_csv(dataset_source, start_date+idx)

        num_points = len(df)
        num_sample = int(num_requests * alpha_rate)
        scale_factor = (num_sample - 1) // num_points + 1

        filtered_requests = []
        for index, row in df.iterrows():
            arrival_time = row['invocation']
            inputlen=min(1024,int(row["ContextTokens"]))
            outputlen=max(3,int(row["GeneratedTokens"]))
            prompt = np.ones(inputlen, dtype=int).tolist()
            for _ in range(scale_factor):
                filtered_requests.append((arrival_time, prompt, inputlen,outputlen))

        model_seed = 42+idx  # Ensure a consistent seed for each model
        rand_inst = random.Random(model_seed)
        sampled = rand_inst.sample(filtered_requests, num_sample)

        sampled_requests.append(sampled)

    generate_workload(workload_infos=workload_info,
                        output_file=output_file,
                        sampled_requests=sampled_requests)
    
def get_prewarm_placement(model_yaml: str,
                           alpha: float,
                           output_file: str,
                           character_output_file: str,
                           request_time: int,
                           request_per_second: float,
                           burstiness: float = -1.0) -> None:
    """
    Get the prewarm placement from the model YAML file.
    """
    num_requests = int(request_time * request_per_second)
    workload_info = get_workloads_info_from_yaml(model_yaml, alpha, False)
    sampled_requests = []
    character = {}
    start_date = max(0, 7 - len(workload_info))
    if start_date == 0:
        raise ValueError("Too many Models! The first model cannot get character!")
    for idx,(model_id_,alpha_rate, dataset_source) in enumerate(workload_info):
        my_day = start_date+idx
        muxserve_id, model_id = model_id_
        prefill_time, decode_time = get_inference_time(model_id)
        means, peaks = get_character_from_csv(dataset_source, INTERVAL, prefill_time, decode_time)
        df = process_csv(dataset_source, day=start_date+idx)

        num_points = len(df)
        num_sample = int(num_requests * alpha_rate)
        scale_factor = (num_sample - 1) // num_points + 1

        filtered_requests = []
        for index, row in df.iterrows():
            arrival_time = row['invocation']
            inputlen=min(1024,int(row["ContextTokens"]))
            outputlen=max(3,int(row["GeneratedTokens"]))
            for _ in range(scale_factor):
                filtered_requests.append((arrival_time, muxserve_id, model_id, inputlen, outputlen))

        model_seed = 42+idx  # Ensure a consistent seed for each model
        rand_inst = random.Random(model_seed)
        sampled = rand_inst.sample(filtered_requests, num_sample)

        sampled_requests.append(sampled)

        # Characteristics Analysis
        # Note that we record characteristics of all intervals on Sunday although the experiments only elapse for an hour.
        num_intervals_per_day = 24 * 3600 // INTERVAL
        num_intervals_per_hour = 3600 // INTERVAL
        if len(peaks) < num_intervals_per_day * 6:
            print(f"Error: trace only has data of {len(peaks)} intervals.")
            exit(1)
        past_peaks = []
        past_avgs = []
        sample_scale = num_sample / len(filtered_requests) / my_day
        for i in range(num_intervals_per_hour):
            sum_peak = 0
            sum_avg = 0
            for day in range(my_day):
                sum_peak += peaks[day * num_intervals_per_day + i]
                sum_avg += means[day * num_intervals_per_day + i]
            past_peaks.append(sum_peak * sample_scale)
            past_avgs.append(sum_avg * sample_scale)
        character[model_id] = (past_avgs, past_peaks)
        print(f"Model {model_id} past_peaks = {past_peaks}, past_avgs = {past_avgs}")
    
    all_requests = sum(sampled_requests, [])
    all_requests_sorted = sorted(all_requests, key=lambda x: x[0])

    # Apply burstiness control if specified
    if burstiness >= 0:
        burst_rng = np.random.default_rng(42)
        all_requests_sorted = apply_burstiness(
            all_requests_sorted, burstiness, burst_rng)
        print(f"Applied burstiness CV={burstiness:.2f} to inter-arrival times")

    print(f"Total requests: {len(all_requests_sorted)}")
    with open(output_file, "wb") as f:
        pickle.dump(all_requests_sorted, f)
    
    print(f"Saving workload characteristics...")
    with open(character_output_file, "wb") as f:
        pickle.dump(character, f)

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    np.random.seed(42)  # For reproducibility
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_type", type=str, default="muxserve", help="Type of trace to generate")
    parser.add_argument("--output_file", type=str, default='', help="Path to the output file")
    parser.add_argument("--character_output_file", type=str, default='', help="Path to the output file of characteristics")
    parser.add_argument("--model_yaml", type=str, help="Path to the model YAML file")
    parser.add_argument("--request_per_second", type=float, default=1, help="Number of requests per second")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha value for the workload generation")
    parser.add_argument("--burstiness", type=float, default=-1.0,
                        help="Coefficient of variation (CV) of inter-arrival times. "
                             "0=regular, 1=Poisson, >1=bursty, <0=keep original trace")
    args = parser.parse_args()

    request_time=3600
    trace_type = args.trace_type
    output_file = args.output_file
    character_output_file = args.character_output_file
    model_yaml = args.model_yaml
    alpha = args.alpha
    request_per_second = args.request_per_second

    burstiness = args.burstiness

    if not output_file:
        if burstiness >= 0:
            output_file = f'workload-{trace_type}-{request_per_second}-{alpha}-cv{burstiness}.pkl'
        else:
            output_file = f'workload-{trace_type}-{request_per_second}-{alpha}.pkl'
        print(f"Using default output file {output_file}.")
    if not character_output_file:
        character_output_file = f'character-{request_per_second}-{alpha}.pkl'
        print(f"Using default character output file {character_output_file}.")

    if trace_type == "muxserve":  
        get_muxserve_placement(model_yaml, alpha, output_file, request_time, request_per_second)
    elif trace_type == "prewarm":
        get_prewarm_placement(model_yaml, alpha, output_file, character_output_file, request_time, request_per_second, burstiness)
    else:
        raise ValueError(f"Unknown trace type: {trace_type}.")
