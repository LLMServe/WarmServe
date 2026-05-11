# WarmServe

WarmServe is a multi-LLM serving system that prewarms models according to workload predictions.

## Requirements

WarmServe is built on top of vLLM 0.6.3.post1 and requires the following versions of packages:
- Python 3.10
- PyTorch 2.4.0
- CUDA 12.1


## Environment Preparation

1. Install vLLM
```
conda create -n vllm python=3.10 -y
conda activate vllm
cd vllm-0.6.3.post1
export SETUPTOOLS_SCM_PRETEND_VERSION="0.6.3.post1"
mkdir vllm/vllm_flash_attn
[Configure your local ssh connection to github to clone repos]
pip install -e .
```

2. Patch PyTorch
```
cd [PATH_TO_ANACONDA]
cd envs/vllm/lib/python3.10/site-packages
patch -p1 < [PATH_TO_WARMSERVE]/pytorch-v2.4.0.patch
```

3. Compile CUDA memory extension
```
cd cuda_memory_extension
python setup.py install
```

## How to Run

### 1. Set model directory prefix

```
export MODEL_PATH=/path/to/models
```

Also edit `vllm-0.6.3.post1/vllm/entrypoints/controller/ModelConfig.py` to match your model layout and dataset paths before running the steps below.

### 2. Prepare model weights

1. Partition model weights

```
python -m vllm.entrypoints.controller.partition_model --model-path=$MODEL_PATH --tensor-parallel-size=1 --pipeline-parallel-size=1
```

The partitioned weights are stored in $MODEL_PATH in sharded manner (e.g., tp1_pp1, tp2_pp1, ...). 

2. Create virtual model replicas

```
python -m vllm.entrypoints.controller.create_virtual_models
```

It creates multiple model replicas according to configurations in vllm-0.6.3.post1/vllm/entrypoints/controller/ModelConfig.py.

### 3. Initialize PrewarmManager

1. Launch Ray Cluster

Set the following environment variables before starting Ray and PrewarmManager:
```
ulimit -n 8192                  # required for many Ray workers
export PYTHONUNBUFFERED=1       # real-time log output
export RAY_DEDUP_LOGS=0         # see per-worker logs
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
```

Then:
```
[On head node]
ray start --head --num-gpus=X
[On other nodes]
ray start --address='[Head IP]:6379' --num-gpus=X --block
```

2. [Optional] Preload model into memory
```
python -m vllm.entrypoints.controller.load_model [--only-create-mapping]
```

Add "--only-create-mapping" to only create mappings to model files end with "_mem" so that they can be directly used by serving endpoints.

3. Launch PrewarmManager
```
python -m vllm.entrypoints.controller.prewarm_manager --use-unified-memory --character-file=[PATH_TO_CHARACTERISTICS_FILE] [--disabled-gpus=0,1,...]
```

Flags:
- `--use-unified-memory`: enable model prewarming and KV cache prewarming.
  - `--disable-kv-prewarm`: disable prewarming models in KV cache of other instances.
  - `--disable-prewarm`: disable prewarming new models based on predictions (keep only the core autoscaler).
- `--disable-all-prewarm`: disable all prewarming; workers are created on demand by the vLLM engine (vanilla vLLM behavior).
- Default (neither `--disable-all-prewarm` nor `--use-unified-memory` is set): device and workers are prewarmed, but models and KV caches are not.
- `--use-loaded-model`: use model weights that have been pre-loaded into memory (requires section 3.2 preload to have run).

The character file is produced by `trace-generator/trace_generator.py` via its `--character_output_file` argument; see [trace-generator/README.md](./trace-generator/README.md).

PrewarmManager exposes a FastAPI endpoint on port 9999 and internally spawns the vLLM api_server subprocesses it needs — no separate api_server launch is required.

After launching, wait for the endpoint to become reachable (`curl http://localhost:9999`), then allow ~60 seconds for initial prewarming before sending workload.

### 4. Evaluation

1. Sanity check: send one request to the PrewarmManager endpoint

```
time curl -X POST http://localhost:9999 \
-H "Content-Type: application/json" \
-d '{
    "id": 0,
    "model": "'$MODEL_PATH'/tp1_pp1/0",
    "prompt": "hello",
    "stream": false
}'
```

2. End-to-End experiment

Generate workloads and the character file in the [trace-generator](./trace-generator) folder. Pass the produced character file to `--character-file` when launching PrewarmManager (section 3.3), then run the request generator:

```
python trace-generator/request_generator.py [PATH_TO_WORKLOAD_PKL]
```

Analyze the results with `result_analyzer`:
```
python vllm-0.6.3.post1/vllm/entrypoints/controller/result_analyzer.py [PATH_TO_GENERATOR_LOG] [PATH_TO_PICKLE_DUMP_FILE]
```


## Code Structure

- [`vllm-0.6.3.post1/`](./vllm-0.6.3.post1/) — vLLM v0.6.3.post1 base. WarmServe additions live in [`vllm/entrypoints/controller/`](./vllm-0.6.3.post1/vllm/entrypoints/controller/):
  - `prewarm_manager.py` — Ray actor orchestrating prewarming. Owns GPU workers, schedules replicas, exposes the FastAPI endpoint on port 9999.
  - `scheduler.py` — prediction-based scheduler over a sliding window.
  - `vmm.py` — CUDA Virtual Memory Manager using the driver API (`cuMemMap` / `cuMemAddressReserve`) for 2MB-granularity blocks.
  - `utils.py` — `MyWorkerWrapper` extending vLLM's Ray worker with prewarming hooks.
  - `ModelConfig.py` — model registry (size, TP config, replicas, dataset paths). Edit before running.
  - `common.py` — shared constants and data classes.
  - `partition_model.py`, `load_model.py`, `create_virtual_models.py`, `prewarm_model.py` — one-shot setup utilities used in the How-to-Run steps.
  - `result_analyzer.py` — post-hoc analysis of experiment logs.
- [`cuda_memory_extension/`](./cuda_memory_extension/) — pybind11 module for portable pinned memory.
- [`trace-generator/`](./trace-generator/) — generates synthetic workloads from Azure/ServeGen/BurstGPT traces (`trace_generator.py`) and replays them against the PrewarmManager endpoint (`request_generator.py`).


## Citation
If you use WarmServe in your research, please cite our [paper](https://arxiv.org/abs/2512.09472).
```
@misc{lou2025warmserveenablingoneformanygpu,
      title={WarmServe: Enabling One-for-Many GPU Prewarming for Multi-LLM Serving}, 
      author={Chiheng Lou and Sheng Qi and Rui Kang and Yong Zhang and Chen Sun and Pengcheng Wang and Bingyang Liu and Xuanzhe Liu and Xin Jin},
      year={2025},
      eprint={2512.09472},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2512.09472}, 
}
```