"""
(1) Set API_SERVER_ADDR to the server address.

(2) Workload should be of type List[Tuple[float, str, int, int]], with each item represents the information of a request:
1. arrival time (in second)
2. model id
3. #input tokens
4. #output tokens

(3) Run python request_generator.py [PATH_TO_WORKLOAD] [REQ_PER_SECOND]
"""

import os
import sys
import pickle
import time
import requests
import random
import aiohttp
import asyncio
import errno
import httpx
import traceback
from random import sample
from openai import AsyncOpenAI

global_prompt = "a"
for _ in range(1024):
    global_prompt += " a"

def get_requests(workload_path):
    muxserve = int(os.getenv('MUXSERVE', '0'))
    if muxserve:
        served_models = os.getenv('MODELS', '').split(',')
        print(f"Served models = {served_models}")
    
    with open(workload_path, 'rb') as f:
        reqs = pickle.load(f)
    
    print(f"Load {len(reqs)} requests in total.")

    mx_time_stamp = 20 * 60     # 20 minutes

    for index, req in enumerate(reqs):
        if req[0] >= mx_time_stamp:
            break
    
    reqs = reqs[:index]

    # Get id
    reqs_ = []
    for req in reqs:
        arrival_time, muxserve_id, model_id, input_tokens, output_tokens = req
        if muxserve:
            if muxserve_id not in served_models:
                continue
            # Remove suffix tpx_ppx
            pos = model_id.rfind("/tp")
            model_id = model_id[:pos]
        reqs_.append((arrival_time, muxserve_id, model_id, input_tokens, output_tokens))
    
    print(f"Get {len(reqs)} requests in {mx_time_stamp} seconds")
    return reqs_, mx_time_stamp

async def run_requests(req, add_mem_suffix: bool, perf_model: str):
    muxserve = int(os.getenv('MUXSERVE', '0'))
    use_openai = int(os.getenv('USE_OPENAI', '0'))
    serverlessllm = int(os.getenv('SERVERLESSLLM', '0'))
    if serverlessllm:
        use_openai = 1

    start_time = time.time()
    base_time = reqs[0][0]

    req_start_time = [0] * len(reqs)
    first_token_time = [0] * len(reqs)
    decode_time = [0] * len(reqs)
    generation_time = [0] * len(reqs)
    num_token = [0] * len(reqs)
    tasks = []

    if use_openai == 1:
        url = os.getenv('API_SERVER_ADDR') + "/v1"
        timeout = httpx.Timeout(timeout=1800)
        custom_httpx_client = httpx.AsyncClient(timeout=timeout)
        client = AsyncOpenAI(
            base_url=url,
            api_key='serverlessllm',
            http_client=custom_httpx_client,
        )
    else:
        conn = aiohttp.TCPConnector(limit=0, ssl=False)
        timeout = aiohttp.ClientTimeout(total=1800)

    for index, req in enumerate(reqs):
        arrival_time, muxserve_id, model_id, input_tokens, output_tokens = req

        if perf_model:
            model_id = perf_model
            input_tokens = 1024
            output_tokens = 100
            arrival_time = base_time
        
        output_tokens = min(output_tokens, 1023)

        if add_mem_suffix:
            pos = model_id.rfind('/')
            model_id = model_id[:pos] + "_mem" + model_id[pos:]
        while True:
            cur_time = time.time()
            if cur_time - start_time >= arrival_time - base_time:
                break
            await asyncio.sleep(0)
        if muxserve:
            print(f"Generate request {[index + 1]} for model {model_id}. MuxServe = {muxserve_id}")
        else:
            print(f"Generate request {[index + 1]} for model {model_id}")
        payload = {'id': index + 1, 'model': model_id, 'prompt': global_prompt[:(max(input_tokens-8,1)*2-1)], 'max_tokens': str(output_tokens), 'stream': 'True'}

        async def post_request(index, payload):
            try:
                while True:
                    try:
                        req_start_time[index] = time.time()
                        first_token_returned = False
                        if serverlessllm == 1:
                            prompt = [{"role": "user", "content": payload['prompt']}]
                            chat = await client.chat.completions.create(
                                model=payload['model'],
                                messages=prompt,
                                max_tokens=int(payload['max_tokens']),
                                temperature=2,
                            )
                            if chat.created is not None:
                                generation_time[index] = time.time()
                                first_token_time[index] = generation_time[index] - chat.created - req_start_time[index]
                                num_generated_tokens = int(payload['max_tokens'])
                                first_token_returned = True
                            else:
                                print(f"Error: request [{index + 1}] does not contain first token time, usage = {chat.usage}. Retry.")
                                continue
                        elif use_openai == 1:
                            prompt = [{"role": "user", "content": payload['prompt']}]
                            chat = await client.chat.completions.create(
                                model=payload['model'],
                                messages=prompt,
                                max_tokens=int(payload['max_tokens']),
                                temperature=2,
                                stream=True,
                                stream_options={"include_usage": True},
                            )
                            async for stream_response in chat:
                                if stream_response.choices[0].delta.content is not None:
                                    if not first_token_returned:
                                        first_token_time[index] = time.time() - req_start_time[index]
                                        first_token_returned = True
                            if stream_response.usage is None:
                                print(f"Error: stream_response final return has null usage")
                                num_generated_tokens = int(payload['max_tokens'])
                            else:
                                num_generated_tokens = stream_response.usage.completion_tokens
                            generation_time[index] = time.time()
                        else:
                            resp_bytes = bytes()
                            async with aiohttp.request('POST', "http://0.0.0.0:9999", json=payload, connector=conn, timeout=timeout) as resp:
                                async for data in resp.content.iter_any():
                                    if not first_token_returned:
                                        first_token_time[index] = time.time() - req_start_time[index]
                                        first_token_returned = True
                                    resp_bytes += data
                            resp_str = resp_bytes.decode('utf-8')
                            pos = resp_str.rfind('#')
                            if pos == -1:
                                print(f"Error: stream_response final return has null usage")
                                num_generated_tokens = int(payload['max_tokens'])
                            else:
                                num_generated_tokens = int(resp_str[pos+1:])
                            generation_time[index] = time.time()
                            if num_generated_tokens != int(payload['max_tokens']):
                                print(f"Warning: Request [{index + 1}] token mismatch. Generated {num_generated_tokens} tokens instead of {int(payload['max_tokens'])}.")
                        if not first_token_returned:
                            print(f"Error: request [{index + 1}] for model {payload['model']} has no return.")
                        if first_token_returned:
                            tot_decode_time = generation_time[index] - req_start_time[index] - first_token_time[index]
                            num_decode_tokens = num_generated_tokens - 1
                            tpot = (tot_decode_time / num_decode_tokens) * 1000.0 if num_decode_tokens >= 1 else -1
                            decode_time[index] = tpot
                            print(f"[{index + 1}]: TTFT = {first_token_time[index]}s, TPOT = {tpot} ms, #Tokens = {num_generated_tokens} ({payload['model']})", flush=True)
                        else:
                            print(f"[{index + 1}]: No Token Response!")
                        break
                    except OSError as e:
                        if e.errno == errno.ECONNRESET:
                            print("Meet exception: Connection Reset By Peer. Retry.")
                        else:
                            exc_info = sys.exc_info()
                            print(f"Request [{index + 1}] gets exception: {e}")
                            if use_openai == 0:
                                print(f"current resp = {resp_bytes}")
                            print("".join(traceback.format_exception(*exc_info)))
                            break
            except Exception as e:
                exc_info = sys.exc_info()
                print(f"Request [{index + 1}] gets exception: {e}")
                if use_openai == 0:
                    print(f"current resp = {resp_bytes}")
                print("".join(traceback.format_exception(*exc_info)))
        tasks.append(asyncio.create_task(post_request(index, payload)))

        if perf_model:
            await tasks[-1]

    print(f"End of generating requests, elapsed = {time.time() - start_time}")

    for task in tasks:
        await task
    
    print(f"All tasks finished, elapsed = {time.time() - start_time}", flush=True)

    if use_openai == 0:
        conn.close()
    # for index in range(len(reqs)):
    #     print(f"[{index + 1}] TTFT = {first_token_time[index]}, TPOT = {decode_time[index]}, Generation time = {generation_time[index] - req_start_time[index]}")

async def single_load_generator(add_mem_suffix: bool):
    init_send = int(os.getenv("INIT_SEND", "0"))
    warm = int(os.getenv("WARM", "0"))
    conn = aiohttp.TCPConnector(limit=0, ssl=False)
    timeout = aiohttp.ClientTimeout(total=3600)

    async def post_request(payload, url):
        try:
            first_token_returned = False
            stime = time.time()
            resp_bytes = bytes()
            async with aiohttp.request('POST', url, json=payload, connector=conn, timeout=timeout) as resp:
                async for data in resp.content.iter_any():
                    if not first_token_returned:
                        first_token_time = time.time() - stime
                        first_token_returned = True
                    resp_bytes += data
            if not first_token_returned:
                print(f"Error: No first token returned")
                first_token_time = time.time() - stime
            resp_str = resp_bytes.decode('utf-8')
            pos = resp_str.rfind('#')
            num_generated_tokens = int(resp_str[pos+1:])
            if not first_token_returned:
                print(f"Error: request for model {model_id} has no return")
            generation_time = time.time()
        except Exception as e:
                exc_info = sys.exc_info()
                print(f"Request gets exception: {e}")
                print("".join(traceback.format_exception(*exc_info)))
        return first_token_time, generation_time - stime - first_token_time, num_generated_tokens
    
    expr_model_list = [
        "/path/to/models/llama-7b/tp1_pp1",
        "/path/to/models/llama-13b/tp2_pp1",
        "/path/to/models/llama-70b/tp4_pp1",
    ]

    ttft_list = []
    tpot_list = []

    num_profile_points = 4
    prompt = 'Write a story in 100 words'

    cur_id = 0
    for model_id in expr_model_list:
        model_id_ = model_id + ("_mem" if add_mem_suffix else "") + "/0"
        payload = {'id': cur_id, 'model': model_id_, 'prompt': prompt, 'stream': 'True'}

        cur_id += 1
        sum_ttft = 0
        sum_tpot = 0
        mn_ttft = 10000
        mn_tpot = 10000
        mx_ttft = 0
        mx_tpot = 0
        if init_send or warm:
            # Send a request first
            ttft, gt, num_tokens = await post_request(payload, "http://0.0.0.0:9999")
            if not warm:
                await asyncio.sleep(10)
        for i in range(num_profile_points):
            print(f"Start measure model {model_id} [{i}]", flush=True)
            ttft, gt, num_tokens = await post_request(payload, "http://0.0.0.0:9999")
            tpot = gt / (num_tokens - 1)
            tpot *= 1000.0
            mn_ttft = min(mn_ttft, ttft)
            mx_ttft = max(mx_ttft, ttft)
            mn_tpot = min(mn_tpot, tpot)
            mx_tpot = max(mx_tpot, tpot)
            sum_ttft += ttft
            sum_tpot += tpot
            print(f"End measure model {model_id} [{i}], ttft = {ttft} s, tpot = {tpot} ms", flush=True)
            if not warm:
                await asyncio.sleep(10)
        ttft = (sum_ttft - mx_ttft - mn_ttft) / (num_profile_points - 2)
        tpot = (sum_tpot - mx_tpot - mn_tpot) / (num_profile_points - 2)
        print(f"Model {model_id}: TTFT = {ttft} s, TPOT = {tpot} ms", flush=True)
        ttft_list.append(ttft)
        tpot_list.append(tpot)
    
    for index, model_id in enumerate(expr_model_list):
        print(f"Model {model_id}: TTFT = {ttft_list[index]} s, TPOT = {tpot_list[index]} ms", flush=True)

if __name__ == '__main__':
    random.seed(42)
    add_mem_suffix = True if int(os.getenv("ADD_MEM_SUFFIX", "0")) == 1 else False
    expr_1_1 = True if int(os.getenv("EXPR_1_1", "0")) == 1 else False
    perf_model = os.getenv("PERF_MODEL", "")
    if expr_1_1:
        asyncio.run(single_load_generator(add_mem_suffix))
        exit(0)
    
    workload_path = sys.argv[1]
    reqs, mx_time_stamp = get_requests(workload_path)

    if perf_model:
        # Only use first three requests
        reqs = reqs[:3]

    count = {}
    for index, req in enumerate(reqs):
        arrival_time, model_id, muxserve_id, input_tokens, output_tokens = req
        if model_id not in count:
            count[model_id] = 1
        else:
            count[model_id] += 1
    count_list = []
    for key, value in count.items():
        count_list.append(value)
    count_list.sort()
    print(f"#Models = {len(count_list)}, Invocation Counts: {count_list}")
    print(f"Start generate {len(reqs)} requests in {mx_time_stamp} seconds")

    asyncio.run(run_requests(reqs, add_mem_suffix, perf_model))
