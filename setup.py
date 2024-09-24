from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize
import numpy
#  python3 setup.py built_ext --inplace

VERSION = '1.0.0'

import os
os.environ['CFLAGS'] = '-O3 -march=native -Wall'


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

docs = [
    'sphinx==5.0.0',
    'sphinx-rtd-theme==0.5.1',
    'sphinxcontrib-youtube==1.0.1',
    'sphinx-rtd-theme==0.5.1',
    'sphinx-design==0.4.1',
    'furo==2023.3.27',
]

cleanrl = [
    'stable_baselines3==2.1.0',
    'tensorboard==2.11.2',
    'torch',
    'tyro==0.8.6',
    'wandb==0.13.7',
]

ray = [
    'ray==2.23.0',
]

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
    'magent': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'pettingzoo==1.19.0',
        'magent==0.2.4',
        # The Magent2 package is broken for now
        #'magent2==0.3.2',
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
        f'gymnasium[mujoco]=={GYMNASIUM_VERSION}',
        'mujoco==2.3.7',  # mujuco > 3 is supported by gymnasium > 1.0
        'moviepy',
    ],
    'nethack': [
        f'gym=={GYM_VERSION}',
        f'gymnasium=={GYMNASIUM_VERSION}',
        'nle>=0.9.1',
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


# These are the environments that PufferLib has made
# compatible with the latest version of Gym/Gymnasium/PettingZoo
# They are included in PufferTank as a default heavy install
# We force updated versions of Gym/Gymnasium/PettingZoo here to
# ensure that users do not have issues with conflicting versions
# when switching to incompatible environments
common = cleanrl + [environments[env] for env in [
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

extension_paths = [
    'pufferlib/environments/ocean/moba/cy_moba',
    'pufferlib/environments/ocean/snake/cy_snake',
    'pufferlib/environments/ocean/pong/cy_pong',
    'pufferlib/environments/ocean/breakout/cy_breakout',
    'pufferlib/environments/ocean/connect4/cy_connect4',
    'pufferlib/environments/ocean/grid/cy_grid',
    'pufferlib/environments/ocean/tripletriad/cy_tripletriad',
    'cytest/cytest',
]

extensions = [Extension(
    path.replace('/', '.'),
    [path + '.pyx'],
    include_dirs=[numpy.get_include(), 'raylib-5.0_linux_amd64/include'],
    library_dirs=['raylib-5.0_linux_amd64/lib'],
    libraries=["raylib"],
    runtime_library_dirs=["raylib-5.0_linux_amd64/lib"],
    extra_compile_args=['-DPLATFORM_DESKTOP'],
) for path in extension_paths]
 
setup(
    name="pufferlib",
    description="PufferAI Library"
    "PufferAI's library of RL tools and utilities",
    long_description_content_type="text/markdown",
    version=VERSION,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy==1.23.3',
        'opencv-python==3.4.17.63',
        'cython>=3.0.0',
        'rich',
        'rich_argparse',
        f'gym<={GYM_VERSION}',
        f'gymnasium<={GYMNASIUM_VERSION}',
        f'pettingzoo<={PETTINGZOO_VERSION}',
        'shimmy[gym-v21]',
        'psutil==5.9.5',
        'pynvml',
        'imageio',
    ],
    extras_require={
        'docs': docs,
        'ray': ray,
        'cleanrl': cleanrl,
        'common': common,
        **environments,
    },
    ext_modules = cythonize([
        "pufferlib/extensions.pyx",
        "c_gae.pyx",
        "pufferlib/environments/ocean/grid/c_grid.pyx",
        "pufferlib/environments/ocean/snake/c_snake.pyx",
        "pufferlib/environments/ocean/moba/c_moba.pyx",
        "pufferlib/environments/ocean/moba/puffernet.pyx",
        "pufferlib/environments/ocean/moba/c_precompute_pathing.pyx",
        *extensions,
    ], 
       #nthreads=6,
       #annotate=True,
       #compiler_directives={'profile': True},# annotate=True
    ),
    include_dirs=[numpy.get_include(), 'raylib-5.0_linux_amd64/include'],
    python_requires=">=3.8",
    license="MIT",
    author="Joseph Suarez",
    author_email="jsuarez@puffer.ai",
    url="https://github.com/PufferAI/PufferLib",
    keywords=["Puffer", "AI", "RL", "Reinforcement Learning"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

#stable_baselines3
#supersuit==3.3.5
#'git+https://github.com/oxwhirl/smac.git',

#curl -L -o smac.zip https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
#unzip -P iagreetotheeula smac.zip 
#curl -L -o maps.zip https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
#unzip maps.zip && mv SMAC_Maps/ StarCraftII/Maps/
