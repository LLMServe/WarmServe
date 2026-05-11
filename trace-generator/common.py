# Inference time of models
# Format: model_id -> (prefill_time, decode_time_per_iteration) (ms)
# Replace `/path/to/models` below with your MODEL_PATH root.
time_dict = {
    "/path/to/models/llama-7b/tp1_pp1": (35, 15),
    "/path/to/models/llama-13b/tp2_pp1": (70, 20),
    "/path/to/models/llama-70b/tp4_pp1": (100, 30),
}