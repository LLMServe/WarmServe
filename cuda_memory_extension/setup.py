# setup.py

from setuptools import setup, Extension
import pybind11
import os

cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
cuda_include = os.path.join(cuda_home, 'include')
cuda_lib64 = os.path.join(cuda_home, 'lib64')

ext_modules = [
    Extension(
        'cuda_allocator',
        ['cuda_allocator.cpp'],
        include_dirs=[
            pybind11.get_include(),
            cuda_include,
        ],
        library_dirs=[cuda_lib64],
        libraries=['cudart'],
        language='c++',
        extra_compile_args=['-std=c++17'],
    ),
]

setup(
    name='cuda_allocator',
    version='0.1',
    description='A Pybind11 extension for CUDA portable pinned memory',
    ext_modules=ext_modules,
)