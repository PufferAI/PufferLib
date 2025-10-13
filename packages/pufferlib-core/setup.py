import os
import sys
import shutil
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Always build extensions
BUILD_EXTENSIONS = True

# Import torch for extensions
try:
    import torch
    from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
    print("Building pufferlib-core with C++/CUDA extensions")
except ImportError:
    print("Error: torch not available. Please install torch first.")
    sys.exit(1)

# Build with DEBUG=1 to enable debug symbols
DEBUG = os.getenv("DEBUG", "0") == "1"

# Compile args
cxx_args = ['-fdiagnostics-color=always']
nvcc_args = []

if DEBUG:
    cxx_args += ['-O0', '-g']
    nvcc_args += ['-O0', '-g']
else:
    cxx_args += ['-O3']
    nvcc_args += ['-O3']

# Extensions setup
torch_sources = ["pufferlib/extensions/pufferlib.cpp"]

# Get torch library path for rpath
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")

# Check if CUDA compiler is available
if shutil.which("nvcc"):
    extension_class = CUDAExtension
    torch_sources.append("pufferlib/extensions/cuda/pufferlib.cu")
    print("Building with CUDA support")
else:
    extension_class = CppExtension
    print("Building with CPU-only support")

# Add rpath for torch libraries
extra_link_args = []
if platform.system() == "Darwin":  # macOS
    extra_link_args.extend([
        f"-Wl,-rpath,{torch_lib_path}",
        "-Wl,-headerpad_max_install_names"
    ])
elif platform.system() == "Linux":  # Linux
    extra_link_args.extend([
        f"-Wl,-rpath,{torch_lib_path}",
        f"-Wl,-rpath,$ORIGIN"
    ])

ext_modules = [
    extension_class(
        "pufferlib._C",
        torch_sources,
        extra_compile_args={
            "cxx": cxx_args,
            "nvcc": nvcc_args,
        },
        extra_link_args=extra_link_args
    ),
]
cmdclass = {"build_ext": BuildExtension}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)