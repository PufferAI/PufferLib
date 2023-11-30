from setuptools import find_packages, setup
from Cython.Build import cythonize
from itertools import chain

VERSION = '0.5.0'

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
GYM_VERSION = '0.21'
PETTINGZOO_VERSION = '1.24.1'
SHIMMY = 'shimmy[gym-v21]'

docs = [
    'sphinx==5.0.0',
    'sphinx-rtd-theme==0.5.1',
    'sphinxcontrib-youtube==1.0.1',
    'sphinx-rtd-theme==0.5.1',
    'sphinx-design==0.4.1',
    'furo==2023.3.27',
]


# These are the environments that PufferLib has made
# compatible with the latest version of Gym/Gymnasium/PettingZoo
# They are included in PufferTank as a default heavy install
# We force updated versions of Gym/Gymnasium/PettingZoo here to
# ensure that users do not have issues with conflicting versions
# when switching to incompatible environments
compatible_environments = {
    'atari': [
        'gymnasium[atari,accept-rom-license]',
        'stable_baselines3==2.1.0',
    ],
    'box2d': [
        'swig==4.1.1',
        'gymnasium[box2d]',
    ],
    'butterfly': [
        'pettingzoo[butterfly]',
    ],
    'classic_control': [
    ],
    'crafter': [
        'crafter==1.8.2',
    ],
    'dm_control': [
        'dm_control==1.0.11',
    ],
    'dm_lab': [
        'gym_deepmindlab==0.1.2',
        'dm_env==1.6',
    ],
    'griddly': [
        'imageio==2.23.0',
        'griddly==1.4.2',
    ],
    'microrts': [
        'ffmpeg==1.4',
        'gym_microrts==0.3.2',
    ],
    'minigrid': [
        'minigrid==2.3.1',
    ],
    'minihack': [
        'minihack==0.1.5',
    ],
    'nethack': [
        'nle==0.9.0',
    ],
    'nmmo': [
        'nmmo',
    ],
    'pokemon_red': [
        'einops==0.6.1',
        'matplotlib',
        'scikit-image==0.21.0',
        'pyboy<2.0.0',
        'hnswlib==0.7.0',
        'mediapy',
        'pandas==2.0.2',
    ],
    'procgen': [
        'procgen==0.10.7',
    ],
}

for env, packages in compatible_environments.items():
    compatible_environments[env] = [
        f'gymnasium=={GYMNASIUM_VERSION}',
        f'gym=={GYM_VERSION}',
        f'pettingzoo=={PETTINGZOO_VERSION}',
        SHIMMY,
        *packages,
    ]

# These environments require specific old versions of 
# Gym/Gymnasium/PettingZoo to work.
incompatible_environments = {
    'avalon': [
        'avalon-rl==1.0.0',
        f'gymnasium=={GYMNASIUM_VERSION}',
        f'pettingzoo=={PETTINGZOO_VERSION}',
    ],
    'magent': [
        'magent==0.2.4',
        'pettingzoo==1.19.0',
        f'gymnasium=={GYMNASIUM_VERSION}',
        f'gym=={GYM_VERSION}',
        # The Magent2 package is broken for now
        #'magent2==0.3.2',
    ],
    'minerl': [
        'gym==0.17.0',
        'minerl==0.4.4',
        # Compatiblity warning with urllib3 and chardet
        'requests==2.31.0',
        f'gymnasium=={GYMNASIUM_VERSION}',
        f'pettingzoo=={PETTINGZOO_VERSION}',
    ],
    'open_spiel': [
        'open_spiel==1.3',
        'pettingzoo==1.19.0',
        f'gymnasium=={GYMNASIUM_VERSION}',
        f'gym=={GYM_VERSION}',
        SHIMMY,
    ],
    #'smac': [
    #    'git+https://github.com/oxwhirl/smac.git',
    #],
    #'stable-retro': [
    #    'git+https://github.com/Farama-Foundation/stable-retro.git',
    #]
}

rllib = [
    'ray[all]==2.0.0',
    'setproctitle==1.1.10',
    'service-identity==21.1.0',
    'pydantic==1.9',
]

cleanrl = [
    'tensorboard==2.11.2',
    'torch',
    'wandb==0.13.7',
    'psutil==5.9.5',
]

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
        'openskill==4.0.0',
        'cython==3.0.0',
    ],
    extras_require={
        'docs': docs,
        'rllib': rllib,
        'cleanrl': cleanrl,
        'compatible-environments': compatible_environments,
        **compatible_environments,
        **incompatible_environments,
    },
    ext_modules = cythonize("pufferlib/extensions.pyx"),
    python_requires=">=3.8",
    license="MIT",
    author="Joseph Suarez",
    author_email="jsuarez@mit.edu",
    url="https://github.com/PufferAI/PufferLib",
    keywords=["Puffer", "AI", "RL"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

#stable_baselines3
#supersuit==3.3.5
#'git+https://github.com/oxwhirl/smac.git',

#curl -L -o smac.zip https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
#unzip -P iagreetotheeula smac.zip 
#curl -L -o maps.zip https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
#unzip maps.zip && mv SMAC_Maps/ StarCraftII/Maps/
