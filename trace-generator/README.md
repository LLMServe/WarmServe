# Trace Generator

## Generate model config

You can automatically generate the model config file according to [ModelConfig](../vllm-0.6.3.post1/vllm/entrypoints/controller/ModelConfig.py).
```
export NUM_NODES = [Number of GPU servers in the cluster]
export NUM_GPUS_PER_NODE = [Number of GPUs per server]
python -m vllm.entrypoints.controller.generate_model_config [SAVE_PATH]
```

## Obtaining trace

We generate workloads according to AzureConv and AzureCode datasets.
We only use the requests on Sunday.

For prewarming system, we additionally generate workload characters to prewarm (`--character_output_file`).
Its format is a dict of `[(model_id, data_source) -> (means, peaks)]`

```
cd trace-generator
python trace_generator.py --trace_type=prewarm --output_file=workload.pkl --character_output_file=character.pkl --model_yaml=models.yaml --request_per_second=1 --alpha=2
```

Parameters:
- trace_type: mux_serve or prewarm
- output_file: Path to the output file
- character_output_file: Path to the output file of model characteristics
- model_yaml: Path to the model YAML file
- request_per_second: Number of requests per second
- alpha: Alpha value for the workload generation

## Generate workloads

For the prewarming system, run the following command to generate requests
```
cd trace-generator
python request_generator [PATH_TO_WORKLOAD]
```

If you are using models preloaded into memory, please configure the environment variable "ADD_MEM_SUFFIX" to true.