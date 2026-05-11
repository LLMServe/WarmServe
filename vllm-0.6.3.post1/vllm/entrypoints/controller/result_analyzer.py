import os
import sys
import pickle
import re
import random
import numpy as np
from random import sample
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import ast

from ModelConfig import ModelList

@dataclass
class Result:
    ttft: float
    tpot: float
    tokens: int
    model: str
    model_id: int
    muxserve_id: str

def process_results(lines: List[str], is_muxserve: bool) -> Dict[int, Result]:
    if is_muxserve:
        pattern = r"\[(\d+)\]: TTFT = (-?[\d\.]+)s, TPOT = (-?[\d\.]+) ms, #Tokens = (\d+) \((.+)\)"
    else:
        pattern = r"\[(\d+)\]: TTFT = (-?[\d\.]+)s, TPOT = (-?[\d\.]+) ms, #Tokens = (\d+) \((.+)/(\d+)\)"
    if is_muxserve:
        gen_pattern = r"Generate request \[(\d+)\] for model (.+). MuxServe = (.+)"
    else:
        gen_pattern = r"Generate request \[(\d+)\] for model (.+)/(\d+)"
    res_dict = {}
    gen_dict = {}

    num = 0
    num_tokens = 0
    sum_ttft = 0
    sum_tpot = 0
    mx_id = 0

    for line in lines:
        match = re.match(pattern, line)
        if match:
            id = int(match.group(1))
            ttft = float(match.group(2))
            tpot = float(match.group(3))
            tokens = int(match.group(4))
            model = match.group(5)
            if is_muxserve:
                model_id = -1
            else:
                model_id = int(match.group(6))

            if "_mem" in model:
                model = model.replace("_mem", "")

            num += 1
            num_tokens += tokens 
            sum_ttft += ttft
            sum_tpot += tpot * tokens

            res_dict[id] = Result(ttft, tpot, tokens, model, model_id, None)
        else:
            match = re.match(gen_pattern, line)
            if match:
                id = int(match.group(1))
                model = match.group(2)
                if is_muxserve:
                    model_id = -1
                    muxserve_id = match.group(3)
                else:
                    model_id = int(match.group(3))
                    muxserve_id = None
                
                if "_mem" in model:
                    model = model.replace("_mem", "")

                gen_dict[id] = (model, model_id, muxserve_id)
                mx_id = max(mx_id, id)
        
    print(f"AVG TTFT = {sum_ttft / num}")
    print(f"AVG TPOT (per token) = {sum_tpot / num_tokens}")
    print(f"AVG tokens/req = {num_tokens / num}")

    if num < mx_id:
        print(f"There are {mx_id-num} requests do not have results. Assume very large TTFT and TPOT.")
    for id in range(1, mx_id + 1):
        if id not in res_dict:
            if id not in gen_dict:
                raise ValueError(f"Error: id {id} not in generation dict and result dict.")
            res_dict[id] = Result(999, 999, 0, gen_dict[id][0], gen_dict[id][1], gen_dict[id][2])
        else:
            res_dict[id].muxserve_id = gen_dict[id][2]

    return res_dict

if __name__ == '__main__':
    random.seed(51)
    is_muxserve = int(os.getenv("MUXSERVE", "0"))

    result_file = sys.argv[1]

    if len(sys.argv) >= 3:
        pickle_file = sys.argv[2]
    else:
        pickle_file = None

    result_handler = open(result_file, 'r')
    lines = result_handler.readlines()
    res_dict = process_results(lines, is_muxserve)
    num_request = len(res_dict.keys())
    result_handler.close()

    model_task_type = {}      # model -> task type
    model_ttfts = {}
    model_tpots = {}
    model_output_tokens = {}
    num_reqs = {}             # task type -> num_reqs
    sum_tokens = {}           # task type -> sum_tokens
    model_sum_tokens = {}     # model -> sum_tokens (expect for fisrt token)
    model_sum_decoding = {}   # model -> sum_decoding_time
    model_sum_tpot = {}       # model -> sum_tpot
    cur_model_index = {}
    counter = 0
    if is_muxserve:
        mapping = {}
    for task_type, model_dict in ModelList.items():
        sum_tokens[task_type] = 0
        num_reqs[task_type] = 0
        for model_id, model_info in model_dict.items():
            # ttft_slo = model_info[1]
            # tpot_slo = model_info[2]
            model_num = model_info[2]
            if model_id not in cur_model_index:
                cur_model_index[model_id] = 0
                cur_index = 0
            else:
                cur_index = cur_model_index[model_id]
            for index in range(model_num):
                model_id_ = model_id + "/" + str(index + cur_index)
                model_task_type[model_id_] = task_type
                model_ttfts[model_id_] = []
                model_tpots[model_id_] = []
                model_output_tokens[model_id_] = []
                if is_muxserve:
                    mapping[f"llm-{counter}"] = model_id_
                counter += 1
            cur_model_index[model_id] += model_num

    # ttft_violation = {}
    # tpot_violation = {}
    # for task_type in ModelList.keys():
    #     ttft_violation[task_type] = 0
    #     tpot_violation[task_type] = 0

    used_models = set()
    for index in range(num_request):
        # Obtain TTFT and TPOT
        id = index + 1
        if id not in res_dict:
            print(f"Error: cannot find result for index {id}")
        else:
            res = res_dict[id]
            if is_muxserve:
                model_id = mapping[res.muxserve_id]
            else:
                model_id = res.model + "/" + str(res.model_id)
            if model_id not in model_task_type:
                print(f"Error: index {id} has model id {model_id} which is not in model_task_type")
            else:
                # Obtain SLO
                used_models.add(model_id)
                task_type = model_task_type[model_id]
                # ttft_slo, tpot_slo = model_slo[model_id]
                # ttft_slo /= 1000
                num_reqs[task_type] += 1
                sum_tokens[task_type] += res.tokens
                if model_id not in model_sum_tokens:
                    model_sum_tokens[model_id] = 0
                    model_sum_decoding[model_id] = 0
                    model_sum_tpot[model_id] = 0
                model_sum_tokens[model_id] += res.tokens - 1
                model_sum_decoding[model_id] += res.tpot * (res.tokens - 1)
                model_sum_tpot[model_id] += res.tpot

                model_ttfts[model_id].append(res.ttft)
                model_tpots[model_id].append(res.tpot)
                model_output_tokens[model_id].append(res.tokens - 1)
    
    used_models = sorted(list(used_models))
    print(f"Num Request = {num_request}")
    print(f"Models = {used_models}")
    model_tails = {}
    for model in used_models:
        ttfts = np.array(model_ttfts[model])
        tpots = np.array(model_tpots[model])
        output_tokens = np.array(model_output_tokens[model])
        avg_ttft = ttfts.mean()
        avg_tpot = tpots.mean()
        avg_tpot_per_token = (tpots * output_tokens).sum() / output_tokens.sum()
        print(f"Model {model} #req = {len(ttfts)} :")
        print(f"AVG_TTFT = {avg_ttft}, AVG_TPOT = {avg_tpot}, AVG_TPOT_PER_TOKEN = {avg_tpot_per_token}")
        # TTFT status
        p99 = np.percentile(ttfts, 99)
        p95 = np.percentile(ttfts, 95)
        p90 = np.percentile(ttfts, 90)
        print(f"p99 = {p99}, p95 = {p95}, p90 = {p90}")
        print("")
        model_tails[model] = (p90, p95, p99)
    
    if pickle_file:
        with open(pickle_file, 'wb') as f:
            pickle.dump((model_ttfts, model_tpots, model_tails), file=f)
