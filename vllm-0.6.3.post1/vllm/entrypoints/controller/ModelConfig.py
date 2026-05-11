'''
Model Config
Format: model_name: (model_size_in_bytes, tp_size, num_replica, data_source)

Before running, either set MODEL_PATH to the root of your partitioned model
weights or edit `model_path` below. Replace every `/path/to/datasets/...`
with the actual trace CSV you intend to use.
'''

import os

model_path = os.getenv("MODEL_PATH", "/path/to/models")

ModelList = {
    "chatbot": {
        os.path.join(model_path, "llama-7b/tp1_pp1"): (13476831232, 1, 2, "/path/to/datasets/AzureLLMInferenceTrace_conv_1week.csv"),
        os.path.join(model_path, "llama-13b/tp2_pp1"): (26032558080, 2, 1, "/path/to/datasets/AzureLLMInferenceTrace_conv_1week.csv"),
        os.path.join(model_path, "llama-70b/tp4_pp1"): (137961209856, 4, 1, "/path/to/datasets/AzureLLMInferenceTrace_conv_1week.csv"),
    },
    # "code": {
    #     os.path.join(model_path, "llama-7b/tp1_pp1"): (13476831232, 1, 1, "/path/to/datasets/AzureLLMInferenceTrace_code_1week.csv"),
    #     os.path.join(model_path, "llama-13b/tp2_pp1"): (26032558080, 2, 1, "/path/to/datasets/AzureLLMInferenceTrace_code_1week.csv"),
    #     os.path.join(model_path, "llama-70b/tp4_pp1"): (137961209856, 4, 1, "/path/to/datasets/AzureLLMInferenceTrace_code_1week.csv"),
    # },
}

'''
Model KV Config
Format: model_name: (num_tokens_per_block, block_size_per_gpu_mb)
llama2-7b: A token costs 0.5MB KV cache [-> llama3-8b: A token costs 0.125MB KV cache]
llama2-13b: A token costs 800KB KV cache
llama2-70b: A token costs 0.3125MB KV cache
Note that both K cache and V cache are included for block_size_per_gpu_mb.
The block_size_per_gpu_mb should satisfy that a single K block (or V block)'s size is the multiple of 2MB.
'''
ModelKVConfig = {
    os.path.join(model_path, "llama-7b/tp1_pp1"): (256, 32),
    os.path.join(model_path, "llama-13b/tp2_pp1"): (256, 100),
    os.path.join(model_path, "llama-70b/tp4_pp1"): (256, 20),
}