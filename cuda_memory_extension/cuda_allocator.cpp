// cuda_allocator.cpp

#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace py = pybind11;

void checkCuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA Error: ") + cudaGetErrorString(err) +
            " in " + file + " at line " + std::to_string(line)
        );
    }
}

#define CHECK_CUDA(val) checkCuda((val), __FILE__, __LINE__)

uintptr_t allocate_portable_pinned(size_t size_in_bytes) {
    void* ptr = nullptr;
    CHECK_CUDA(cudaHostAlloc(&ptr, size_in_bytes, cudaHostAllocPortable));
    return reinterpret_cast<uintptr_t>(ptr);
}

void free_portable_pinned(uintptr_t ptr_address) {
    void* ptr = reinterpret_cast<void*>(ptr_address);
    CHECK_CUDA(cudaFreeHost(ptr));
}

uintptr_t register_pinned_memory(uintptr_t ptr_address, size_t size) {
    cudaError_t err = cudaHostRegister(reinterpret_cast<void*>(ptr_address), size, cudaHostRegisterMapped);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaHostRegister failed: " + std::string(cudaGetErrorString(err)));
    }

    void* host_ptr = reinterpret_cast<void*>(ptr_address);
    void* device_ptr = nullptr;
    err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaHostGetDevicePointer failed: " + std::string(cudaGetErrorString(err)));
    }
    return reinterpret_cast<uintptr_t>(device_ptr);
}

void unregister_pinned_memory(uintptr_t ptr_address) {
    cudaHostUnregister(reinterpret_cast<void*>(ptr_address));
}

PYBIND11_MODULE(cuda_allocator, m) {
    m.def("allocate_portable_pinned", &allocate_portable_pinned, 
          "Allocates portable pinned host memory.",
          py::arg("size_in_bytes"));

    m.def("free_portable_pinned", &free_portable_pinned, 
          "Frees portable pinned host memory.",
          py::arg("ptr_address"));
    
    m.def("register_pinned_memory", &register_pinned_memory, 
          "Register pinned memory.",
          py::arg("ptr_address"),
          py::arg("size"));
    
    m.def("unregister_pinned_memory", &unregister_pinned_memory, 
          "Unregister pinned memory.",
          py::arg("ptr_address"));
}