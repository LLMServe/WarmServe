import torch
import ctypes
from ctypes import POINTER, c_void_p, c_size_t, c_uint64, c_int, byref, c_uint
import time
import cupy as cp
import numpy as np
from typing import Dict, Tuple, Optional, List

class CUDAVMMPool:
    def __init__(self):
        self._init_cuda_driver()
        self.pool_size = 0
        self.block_size = 0
        self.device_id = -1
        self.granularity = 0
        self.context = None
        self.cu_device = None
        self.physical_handles: List[ctypes.c_ulonglong] = []
        self.pool_tensor = None
        self.initialized = False
    
    def _init_cuda_driver(self):
        """Initialize CUDA Driver API"""
        cuda = ctypes.CDLL("libcuda.so.1")
        
        cuda.cuInit.argtypes = [c_uint]
        cuda.cuInit.restype = c_int
        
        cuda.cuDeviceGet.argtypes = [POINTER(c_int), c_int]
        cuda.cuDeviceGet.restype = c_int
        
        cuda.cuMemAddressReserve.argtypes = [POINTER(c_uint64), c_size_t, c_size_t, c_uint64, c_uint64]
        cuda.cuMemAddressReserve.restype = c_int
        
        cuda.cuDeviceGetAttribute.argtypes = [POINTER(c_int), c_int, c_int]
        cuda.cuDeviceGetAttribute.restype = c_int

        cuda.cuMemSetAccess.argtypes = [c_uint64, c_size_t, c_void_p, c_size_t]
        cuda.cuMemSetAccess.restype = c_int

        cuda.cuMemMap.argtypes = [c_uint64, c_size_t, c_uint64, c_uint64, c_uint64]
        cuda.cuMemMap.restype = c_int

        cuda.cuMemUnmap.argtypes = [c_uint64, c_size_t]
        cuda.cuMemUnmap.restype = c_int

        cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cuda.cuCtxGetCurrent.restype = ctypes.c_int

        cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        cuda.cuCtxSetCurrent.restype = ctypes.c_int

        functions = [
            "cuInit", "cuDeviceGet", "cuCtxGetCurrent", "cuCtxSetCurrent", "cuDevicePrimaryCtxRetain", 
            "cuCtxPushCurrent_v2", "cuCtxPopCurrent_v2",
            "cuMemCreate", "cuMemRelease", "cuMemAddressReserve", 
            "cuMemAddressFree", "cuMemMap", "cuMemUnmap", 
            "cuMemSetAccess", "cuMemGetAllocationGranularity",
            "cuMemcpyDtoD_v2", "cuMemGetAddressRange_v2",
            "cuDevicePrimaryCtxRelease_v2", "cuMemset_v2"
        ]
    
        self.cuda_lib = cuda
        for func_name in functions:
            if hasattr(self.cuda_lib, func_name):
                setattr(self, func_name, getattr(self.cuda_lib, func_name))
        
        self.cuInit(0)
    
    def _check_cuda(self, result: int, msg: str = "CUDA Error"):
        if result != 0:
            raise RuntimeError(f"{msg}: {result}")
    
    def _align_up(self, size: int, alignment: int) -> int:
        return (size + alignment - 1) // alignment * alignment
    
    def get_current_context(self):
        context = ctypes.c_void_p()
        self._check_cuda(self.cuCtxGetCurrent(ctypes.byref(context)))
        return context.value

    def set_current_context(self, context: int):
        ctx_ptr = ctypes.c_void_p(context)
        self._check_cuda(self.cuCtxSetCurrent(ctx_ptr))
    
    def initialize(self, pool_size_gb: float, block_size_mb: float = 2, device_id: int = 0):
        """
        Initialize memory pool: create all physical blocks and map into a large tensor
        
        Args:
            pool_size_gb: pool size (GB)
            block_size_mb: block size (MB)
            device: CUDA device ID
        """
        if self.initialized:
            print("Pool already initialized")
            return
        
        self.device_id = device_id
        self.pool_size = int(pool_size_gb * 1024**3)
        self.block_size = int(block_size_mb * 1024**2)
        
        self.cu_device = ctypes.c_int()
        self.context = ctypes.c_void_p()
        
        self._check_cuda(self.cuDeviceGet(ctypes.byref(self.cu_device), device_id))
        self._check_cuda(self.cuDevicePrimaryCtxRetain(ctypes.byref(self.context), self.cu_device))
        
        try:
            # Obtain memory granularity
            prop = self._create_alloc_prop()
            granularity = ctypes.c_size_t()
            self._check_cuda(self.cuMemGetAllocationGranularity(
                ctypes.byref(granularity), ctypes.byref(prop), 0))
            self.granularity = granularity.value
            
            # Align up block_size and pool_size
            self.block_size = self._align_up(self.block_size, self.granularity)
            aligned_pool_size = self._align_up(self.pool_size, self.granularity)
            self.pool_size = aligned_pool_size
            
            # Calculate block number
            num_blocks = aligned_pool_size // self.block_size
            if num_blocks == 0:
                raise ValueError(f"Block size {block_size_mb}MB too large for pool size {pool_size_gb}GB")
            
            print(f"Creating {num_blocks} blocks of {self.block_size/1024**2:.1f}MB each...")
            
            self.access_desc = self._create_access_desc()
            
            # create and map all blocks
            self._create_all_blocks(num_blocks, prop)
            
            self.initialized = True
            print(f"VMM Pool initialized: {pool_size_gb:.1f}GB with {num_blocks} physical blocks on device {device_id}")
            
        finally:
            pass
        
        return num_blocks

    def init_virtual_tensors(self, num_virtual_tensors: int):
        virtual_tensors = []
        self.va_ptrs = []
        for i in range(num_virtual_tensors):
            # Reserve virtual address space
            va_ptr = ctypes.c_ulonglong()
            self._check_cuda(self.cuMemAddressReserve(
                ctypes.byref(va_ptr), self.pool_size, 0, ctypes.c_ulonglong(0), 0))
            
            # Preform map and unmap to make the va_ptr available when creating tensors
            self._check_cuda(self.cuMemMap(va_ptr, self.block_size, 0, self.physical_handles[0], 0),
                f"Mapping a initial block for virtual tensor {i}")
            
            tensor = self._create_pool_tensor(self.pool_size, va_ptr)
            virtual_tensors.append(tensor)
            self.va_ptrs.append(va_ptr)

            self._check_cuda(self.cuMemUnmap(va_ptr, self.block_size),
                f"UnMapping a initial block for virtual tensor {i}")
        
        return virtual_tensors
    
    def map_blocks(self, slot_id: int, offset: int, num_blocks: int, block_ids: np.array, sync_signal: List[int], kv: bool, slow_event):
        va_ptr = self.va_ptrs[slot_id].value + offset
        if not kv:
            for i in range(num_blocks):
                block_id = block_ids[i]
                map_va = ctypes.c_ulonglong(va_ptr)
                self._check_cuda(self.cuMemMap(map_va, self.block_size, 0, self.physical_handles[block_id], 0),
                    f"Mapping block {i}")
                if sync_signal:
                    self._set_memory_access(map_va, self.block_size)
                    sync_signal[0] = (i + 1) * self.block_size
                va_ptr += self.block_size
        else:
            # Allocate K cache and V cache at the same speed
            va_ptr_k = va_ptr
            va_ptr_v = va_ptr + (num_blocks // 2) * self.block_size
            v_start_pos = num_blocks // 2
            counter = 0
            for i in range(num_blocks // 2):
                block_id_k = block_ids[i]
                block_id_v = block_ids[i + v_start_pos]
                map_va_k = ctypes.c_ulonglong(va_ptr_k)
                map_va_v = ctypes.c_ulonglong(va_ptr_v)
                self._check_cuda(self.cuMemMap(map_va_k, self.block_size, 0, self.physical_handles[block_id_k], 0),
                    f"Mapping K block {i}")
                self._check_cuda(self.cuMemMap(map_va_v, self.block_size, 0, self.physical_handles[block_id_v], 0),
                    f"Mapping V block {i}")
                if sync_signal:
                    self._set_memory_access(map_va_k, self.block_size)
                    self._set_memory_access(map_va_v, self.block_size)
                    sync_signal[0] = (i + 1) * self.block_size
                va_ptr_k += self.block_size
                va_ptr_v += self.block_size
                counter += 1
                if slow_event.is_set():
                    if counter >= 10:
                        # Sleep 0.01 second after mapping 20MB blocks
                        time.sleep(0.01)
                        counter = 0
            if num_blocks % 2 == 1:
                # Map the additional V block
                block_id_v = block_ids[num_blocks-1]
                map_va_v = ctypes.c_ulonglong(va_ptr_v)
                self._check_cuda(self.cuMemMap(map_va_v, self.block_size, 0, self.physical_handles[block_id_v], 0),
                    f"Mapping additional block")
                if sync_signal:
                    self._set_memory_access(map_va_v, self.block_size)
            if sync_signal:
                sync_signal[0] = num_blocks * self.block_size
        if not sync_signal:
            # Set memory access once
            self._set_memory_access(ctypes.c_ulonglong(self.va_ptrs[slot_id].value + offset), num_blocks * self.block_size)
    
    def unmap_blocks(self, slot_id: int, offset: int, size: int):
        va_ptr = ctypes.c_ulonglong(self.va_ptrs[slot_id].value + offset)
        self._check_cuda(self.cuMemUnmap(va_ptr, size),
                f"UnMapping VA")
    
    def _create_alloc_prop(self):
        class CUmemLocation(ctypes.Structure):
            _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]
        
        class CUmemAllocationProp(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_int),
                ("requestedHandleTypes", ctypes.c_int), 
                ("location", CUmemLocation),
                ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags", ctypes.c_ubyte * 64)
            ]
        
        prop = CUmemAllocationProp()
        ctypes.memset(ctypes.byref(prop), 0, ctypes.sizeof(prop))
        prop.type = 1  # CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = 1  # CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self.device_id
        return prop
    
    def _create_all_blocks(self, num_blocks: int, prop):
        self.physical_handles = []
        
        for i in range(num_blocks):
            try:
                # Create physical blocks
                handle = ctypes.c_ulonglong()
                self._check_cuda(self.cuMemCreate(
                    ctypes.byref(handle), self.block_size, ctypes.byref(prop), 0),
                    f"Creating physical block {i}")
                self.physical_handles.append(handle)
            except Exception as e:
                print(f"Failed to create block {i}: {e}")
                raise e
        
        print(f"Successfully created {len(self.physical_handles)} physical blocks")

    def _create_access_desc(self):
        class CUmemLocation(ctypes.Structure):
            _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]
        
        class CUmemAccessDesc(ctypes.Structure):
            _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]
        
        access_desc = CUmemAccessDesc()
        access_desc.location.type = 1  # CU_MEM_LOCATION_TYPE_DEVICE
        access_desc.location.id = self.device_id
        access_desc.flags = 3  # CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        return access_desc

    def _set_memory_access(self, va_ptr: ctypes.c_ulonglong, size: int):
        self._check_cuda(self.cuMemSetAccess(va_ptr, size, ctypes.byref(self.access_desc), 1))
    
    def _create_pool_tensor(self, pool_size: int, va_ptr: ctypes.c_ulonglong):
        # Create memory region
        unowned_mem = cp.cuda.UnownedMemory(va_ptr.value, pool_size, None, self.device_id)
        mem_ptr = cp.cuda.MemoryPointer(unowned_mem, 0)
        
        # Create CuPy array
        cupy_array = cp.ndarray((pool_size // 2,), dtype=cp.float16, memptr=mem_ptr)
        pool_tensor = torch.as_tensor(cupy_array, device=f'cuda:{self.device_id}')

        return pool_tensor
        
if __name__ == "__main__":
    print(f"-----VMM Test-----")
    free_gpu_memory = 32 * 1024 * 1024 * 1024
    device_id = 0
    pool = CUDAVMMPool()
    num_blocks = pool.initialize(free_gpu_memory / (2**30), 2, device_id)
    num_slots = 30
    tensors = pool.init_virtual_tensors(num_slots)
    print(f"my tensors = {[tensor.data_ptr() for tensor in tensors]}")
    for i in range(3):
        blocks = np.random.randint(num_blocks, size=(num_blocks,), dtype=int)
        stime = time.time()
        pool.map_blocks(i, 0, num_blocks, blocks, None)
        print(f"map {num_blocks} blocks time cost = {(time.time() - stime) * 1000.0} ms")
    len = 1024 * 1024 * 4
    print(tensors[0][:len].cpu())
    for i in range(3):
        stime = time.time()
        pool.unmap_blocks(i, 0, free_gpu_memory)
        print(f"unmap {num_blocks} blocks time cost = {(time.time() - stime) * 1000.0} ms")