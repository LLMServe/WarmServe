import os
import time
import argparse

from vllm.executor.ray_utils import ray

model_path = os.getenv("MODEL_PATH", "")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prewarm-model-name', type=str, default=f'{model_path}/tp1_pp1')
    parser.add_argument('--prewarm-model-tp-size', type=int, default=1)
    parser.add_argument('--prewarm-model-pp-size', type=int, default=1)

    args = parser.parse_args()

    ray.init()
    prewarm_manager = ray.get_actor("prewarm_manager", namespace="prewarm")
    world_size = args.prewarm_model_tp_size * args.prewarm_model_pp_size
    worker_ids = range(world_size)
    print(f"Start to prewarm model {args.prewarm_model_name} for workers {worker_ids}")
    stime = time.time()

    ray.get(prewarm_manager.prewarm_model_sync.remote(args.prewarm_model_name,
                                                      args.prewarm_model_tp_size,
                                                      args.prewarm_model_pp_size,
                                                      worker_ids))
    print(f"Prewarm Complete. Time Cost = {'%.1f' % (time.time() - stime)} seconds")