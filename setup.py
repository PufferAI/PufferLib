#TODO:
# --no-build-isolation for 5090
# Make c and torch compile at the same time
# CUDA_VISIBLE_DEVICES=None LD_PRELOAD=$(gcc -print-file-name=libasan.so) python3.12 -m pufferlib.clean_pufferl eval --train.device cpu
'''
Pain points for docs:
    - Build in C first
    - Make sure obs types match in C and python
    - Getting obs and action spaces and types correct
    - Double check obs are not zero
    - Correct reset behavior
    - Make sure rewards look correct
    - don't forget params/init in binding
    - Use debug mode to catch segaults
    - TODO: Add check on num agents vs obs shape!!
'''


from setuptools import find_packages, find_namespace_packages, setup, Extension
import numpy
import os
import glob
import urllib.request
import zipfile
import tarfile
import platform
import shutil

from setuptools.command.build_ext import build_ext
from torch.utils import cpp_extension
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)


VERSION = "2.0.6"

# Build with DEBUG=1 to enable debug symbols
DEBUG = os.getenv("DEBUG", "0") == "1"

# Build raylib for your platform
RAYLIB_URL = 'https://github.com/raysan5/raylib/releases/download/5.5/'
RAYLIB_NAME = 'raylib-5.5_macos' if platform.system() == "Darwin" else 'raylib-5.5_linux_amd64'
RLIGHTS_URL = 'https://raw.githubusercontent.com/raysan5/raylib/refs/heads/master/examples/shaders/rlights.h'

def download_raylib(platform, ext):
    if not os.path.exists(platform):
        urllib.request.urlretrieve(RAYLIB_URL + platform + ext, platform + ext)
        if ext == '.zip':
            with zipfile.ZipFile(platform + ext, 'r') as zip_ref:
                zip_ref.extractall()
        else:
            with tarfile.open(platform + ext, 'r') as tar_ref:
                tar_ref.extractall()

        os.remove(platform + ext)
        urllib.request.urlretrieve(RLIGHTS_URL, platform + '/include/rlights.h')

download_raylib('raylib-5.5_webassembly', '.zip')

# Shared compile args for all platforms
extra_compile_args = [
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
    '-DPLATFORM_DESKTOP',
]
extra_link_args = [
    '-fwrapv'
]
cxx_args = [
    '-fdiagnostics-color=always',
]
nvcc_args = []

if DEBUG:
    extra_compile_args += [
        '-O0',
        '-g',
        '-fsanitize=address,undefined,bounds,pointer-overflow,leak',
        '-fno-omit-frame-pointer',
    ]
    extra_link_args += [
        '-g',
        '-fsanitize=address,undefined,bounds,pointer-overflow,leak',
    ]
    cxx_args += [
        '-O0',
        '-g',
    ]
    nvcc_args += [
        '-O0',
        '-g',
    ]
else:
    extra_compile_args += [
        '-O2',
        '-flto',
    ]
    extra_link_args += [
        '-O2',
    ]
    cxx_args += [
        '-O3',
    ]
    nvcc_args += [
        '-O3',
    ]

system = platform.system()
if system == 'Linux':
    extra_compile_args += [
        '-Wno-alloc-size-larger-than',
        '-Wno-implicit-function-declaration',
        '-fmax-errors=3',
    ]
    extra_link_args += [
        '-Bsymbolic-functions',
    ]
    download_raylib('raylib-5.5_linux_amd64', '.tar.gz')
elif system == 'Darwin':
    extra_compile_args += [
        '-Wno-error=int-conversion',
        '-Wno-error=incompatible-function-pointer-types',
        '-Wno-error=implicit-function-declaration',
    ]
    extra_link_args += [
        '-framework', 'Cocoa',
        '-framework', 'OpenGL',
        '-framework', 'IOKit',
    ]
    download_raylib('raylib-5.5_macos', '.tar.gz')
else:
    raise ValueError(f'Unsupported system: {system}')

# Default Gym/Gymnasium/PettingZoo versions
# Gym:
# - 0.26 still has deprecation warnings and is the last version of the package
# - 0.25 adds a breaking API change to reset, step, and render_modes
# - 0.24 is broken
# - 0.22-0.23 triggers deprecation warnings by calling its own functions
# - 0.21 is the most stable version
# - <= 0.20 is missing dict methods for gym.spaces.Dict
# - 0.18-0.21 require setuptools<=65.5.0

GYMNASIUM_VERSION = '0.29.1'
GYM_VERSION = '0.23'
PETTINGZOO_VERSION = '1.24.1'

environments = {
    'avalon': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'avalon-rl==1.0.0',
    ],
    'atari': [
        f'gym=={GYM_VERSION}',
        f'gymnasium[accept-rom-license]=={GYMNASIUM_VERSION}',
        'ale_py==0.9.0',
    ],
    'box2d': [
        f'gym=={GYM_VERSION}',
        f'gymnasium[box2d]=={GYMNASIUM_VERSION}',
        'swig==4.1.1',
    ],
    'bsuite': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'bsuite==0.3.5',
    ],
    'butterfly': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        f'pettingzoo[butterfly]=={PETTINGZOO_VERSION}',
    ],
    'classic_control': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
    ],
    'crafter': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'crafter==1.8.3',
    ],
    'craftax': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'craftax',
    ],
    'dm_control': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'dm_control==1.0.11',
    ],
    'dm_lab': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'gym_deepmindlab==0.1.2',
        'dm_env==1.6',
    ],
    'griddly': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'griddly==1.6.7',
        'imageio',
    ],
    'kinetix': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'kinetix-env',
    ],
    'magent': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'pettingzoo==1.19.0',
        'magent==0.2.4',
        # The Magent2 package is broken for now
        #'magent2==0.3.2',
    ],
    'metta': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'omegaconf',
        'hydra-core',
        'duckdb',
    ],
    'microrts': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'ffmpeg==1.4',
        'gym_microrts==0.3.2',
    ],
    'minerl': [
        'gym==0.17.0',
        f'gymnasium=={GYMNASIUM_VERSION}',
        #'git+https://github.com/minerllabs/minerl'
        # Compatiblity warning with urllib3 and chardet
        #'requests==2.31.0',
    ],
    'minigrid': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'minigrid==2.3.1',
    ],
    'minihack': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'minihack==0.1.5',
    ],
    'mujoco': [
        f'gymnasium[mujoco]==1.0.0',
        'moviepy',
    ],
    'nethack': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'nle==0.9.1',
    ],
    'nmmo': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        f'pettingzoo=={PETTINGZOO_VERSION}',
        'nmmo>=2.1',
    ],
    'open_spiel': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'open_spiel==1.3',
        'pettingzoo==1.19.0',
    ],
    'pokemon_red': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'pokegym>=0.2.0',
        'einops==0.6.1',
        'matplotlib',
        'scikit-image',
        'pyboy<2.0.0',
        'hnswlib==0.7.0',
        'mediapy',
        'pandas==2.0.2',
        'pettingzoo',
        'websockets',
    ],
    'procgen': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'procgen-mirror==0.10.7', # Procgen mirror for 3.11 and 3.12 support
        # Note: You need glfw==2.7 after installing for some torch versions
    ],
    #'smac': [
    #    'git+https://github.com/oxwhirl/smac.git',
    #],
    #'stable-retro': [
    #    'git+https://github.com/Farama-Foundation/stable-retro.git',
    #]
    'slimevolley': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'slimevolley==0.1.0',
    ],
    'vizdoom': [
        'vizdoom==1.2.3',
    ],
}

docs = [
    'sphinx==5.0.0',
    'sphinx-rtd-theme==0.5.1',
    'sphinxcontrib-youtube==1.0.1',
    'sphinx-rtd-theme==0.5.1',
    'sphinx-design==0.4.1',
    'furo==2023.3.27',
]

train = [
    'stable_baselines3==2.1.0',
    'tensorboard==2.11.2',
    'torch',
    'tyro==0.8.6',
    'wandb==0.19.1',
    'scipy',
    'pyro-ppl',
    'neptune',
    'heavyball',
]

ray = [
    'ray==2.23.0',
]

# These are the environments that PufferLib has made
# compatible with the latest version of Gym/Gymnasium/PettingZoo
# They are included in PufferTank as a default heavy install
# We force updated versions of Gym/Gymnasium/PettingZoo here to
# ensure that users do not have issues with conflicting versions
# when switching to incompatible environments
common = train + [environments[env] for env in [
    'atari',
    #'box2d',
    'bsuite',
    #'butterfly',
    'classic_control',
    'crafter',
    'dm_control',
    'dm_lab',
    'griddly',
    'microrts',
    'minigrid',
    'minihack',
    'nethack',
    'nmmo',
    'pokemon_red',
    'procgen',
    'vizdoom',
]]

# Extensions 
class BuildExt(build_ext):
    def run(self):
        # Propagate any build_ext options (e.g., --inplace, --force) to subcommands
        build_ext_opts = self.distribution.command_options.get('build_ext', {})
        if build_ext_opts:
            # Copy flags so build_torch and build_c respect inplace/force
            self.distribution.command_options['build_torch'] = build_ext_opts.copy()
            self.distribution.command_options['build_c'] = build_ext_opts.copy()

        # Run the torch and C builds (which will handle copying when inplace is set)
        self.run_command('build_torch')
        self.run_command('build_c')

class CBuildExt(build_ext):
    def run(self, *args, **kwargs):
        self.extensions = [e for e in self.extensions if e.name != "pufferlib._C"]
        super().run(*args, **kwargs)

class TorchBuildExt(cpp_extension.BuildExtension):
    def run(self):
        self.extensions = [e for e in self.extensions if e.name == "pufferlib._C"]
        super().run()

RAYLIB_A = f'{RAYLIB_NAME}/lib/libraylib.a'
INCLUDE = [numpy.get_include(), 'raylib/include']
extension_kwargs = dict(
    include_dirs=INCLUDE,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    extra_objects=[RAYLIB_A],
)

# Put C env names here. PufferLib will look for
# pufferlib/ocean/<name>/binding.c
c_extensions_names = [
    'gpudrive',
    'squared',
    'pong',
    'boids',
    'breakout',
    'enduro',
    'blastar',
    'grid',
    'nmmo3',
    'tactical',
    'connect4',
    'go',
    'cartpole'
]

# TODO: Include other C files so rebuild is auto?
c_extension_paths = glob.glob('pufferlib/ocean/**/binding.c', recursive=True)
c_extensions = [
    Extension(
        path.rstrip('.c').replace('/', '.'),
        sources=[path],
        **extension_kwargs,
    )
    for path in c_extension_paths
]
c_extension_paths = [os.path.join(*path.split('/')[:-1]) for path in c_extension_paths]

# Check if CUDA compiler is available. You need cuda dev, not just runtime.
torch_sources = [
    "pufferlib/extensions/pufferlib.cpp",
]
if shutil.which("nvcc"):
    extension = CUDAExtension
    torch_sources.append("pufferlib/extensions/cuda/pufferlib.cu")
else:
    extension = CppExtension

torch_extensions = [
   extension(
        "pufferlib._C",
        torch_sources,
        extra_compile_args = {
            "cxx": cxx_args,
            "nvcc": nvcc_args,
        }
    ),
]

# Prevent Conda from injecting garbage compile flags
from distutils.sysconfig import get_config_vars
cfg_vars = get_config_vars()
for key in ('CC', 'CXX', 'LDSHARED'):
    if cfg_vars[key]:
        cfg_vars[key] = cfg_vars[key].replace('-B /root/anaconda3/compiler_compat', '')
        cfg_vars[key] = cfg_vars[key].replace('-pthread', '')
        cfg_vars[key] = cfg_vars[key].replace('-fno-strict-overflow', '')

for key, value in cfg_vars.items():
    if value and '-fno-strict-overflow' in str(value):
        cfg_vars[key] = value.replace('-fno-strict-overflow', '')

setup(
    name="pufferlib",
    description="PufferAI Library"
    "PufferAI's library of RL tools and utilities",
    long_description_content_type="text/markdown",
    version=VERSION,
    packages=find_namespace_packages() + find_packages() + c_extension_paths + ['pufferlib/extensions'],
    package_data={
        "pufferlib": [RAYLIB_NAME + '/lib/libraylib.a']
    },
    include_package_data=True,
    install_requires=[
        'numpy',
        'opencv-python==3.4.17.63',
        'rich',
        'rich_argparse',
        f'gym<={GYM_VERSION}',
        f'gymnasium<={GYMNASIUM_VERSION}',
        f'pettingzoo<={PETTINGZOO_VERSION}',
        'torch',
        'shimmy[gym-v21]',
        'psutil==5.9.5',
        'pynvml',
        'imageio',
        'setuptools'
    ],
    extras_require={
        'docs': docs,
        'ray': ray,
        'train': train,
        'common': common,
        **environments,
    },
    ext_modules = c_extensions + torch_extensions,
    cmdclass={
        "build_ext": BuildExt,
        "build_torch": TorchBuildExt,
        "build_c": CBuildExt,
    },
    include_dirs=[numpy.get_include(), RAYLIB_NAME + '/include'],
    python_requires=">=3.9",
    license="MIT",
    author="Joseph Suarez",
    author_email="jsuarez@puffer.ai",
    url="https://github.com/PufferAI/PufferLib",
    keywords=["Puffer", "AI", "RL", "Reinforcement Learning"],
    entry_points={
        'console_scripts': [
            'puffer = pufferlib.pufferl:main',
        ],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
#stable_baselines3
#supersuit==3.3.5
#'git+https://github.com/oxwhirl/smac.git',

#curl -L -o smac.zip https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
#unzip -P iagreetotheeula smac.zip 
#curl -L -o maps.zip https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
#unzip maps.zip && mv SMAC_Maps/ StarCraftII/Maps/
