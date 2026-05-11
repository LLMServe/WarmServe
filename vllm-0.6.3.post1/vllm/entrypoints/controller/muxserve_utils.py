import sys
import subprocess

def create_muxserve_engine(model, tensor_parallel_size, cur_device_pos, mps_percent, gpu_memory_utilization, max_num_seqs, engine_id):
    # max_num_seqs = 32
    
    stdout_file_name = f"engine_stdout{engine_id}.out"
    stderr_file_name = f"engine_stderr{engine_id}.out"
    f_out = open(stdout_file_name, 'w')
    f_err = open(stderr_file_name, 'w')
    env = {
        "CUDA_VISIBLE_DEVICES": ",".join([str(x) for x in range(cur_device_pos, cur_device_pos + tensor_parallel_size)]),
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(mps_percent),
        "VLLM_LOGGING_LEVEL": "DEBUG",
        "MUXSERVE": "1",
    }
    args = f"--model {model} --max-model-len 2048 --max-num-batched-tokens 100000 --max-num-seqs {max_num_seqs} --tensor-parallel-size {tensor_parallel_size} --pipeline-parallel-size 1 --dtype float16 --enforce-eager --block-size 256 --host 0.0.0.0 --port {5050+engine_id} --gpu-memory-utilization {gpu_memory_utilization} --trust-remote-code --disable-frontend-multiprocessing"
    args = args.split()
    process = subprocess.Popen(
        [sys.executable, '-m', 'vllm.entrypoints.openai.api_server'] + args,
        env=env,
        stdin=subprocess.PIPE,
        stdout=f_out,
        stderr=f_err,
        text=True,
    )

    f_out.close()
    f_err.close()

    print(f"Started an engine for model {model} with mps_percentage = {mps_percent}")

    return process