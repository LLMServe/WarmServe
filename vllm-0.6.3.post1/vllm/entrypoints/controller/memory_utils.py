import ctypes
import torch
import numpy as np

def get_shm_name(model_name: str, tp_rank: int):
    shm_name = "model-" + model_name + "-" + str(tp_rank)
    return shm_name.replace("/", "-")

def wrap_ptr_to_tensor(ptr_address: int, size_in_bytes: int, torch_dtype = torch.float16):
    np_dtype = getattr(np, str(torch_dtype).split('.')[-1])
    item_size = np.dtype(np_dtype).itemsize
    shape = (size_in_bytes // item_size,)
    
    numpy_array = np.ctypeslib.as_array(
        (ctypes.c_char * size_in_bytes).from_address(ptr_address)
    ).view(np_dtype).reshape(shape)
    
    tensor = torch.from_numpy(numpy_array)
    return tensor