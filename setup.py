from pdb import set_trace as T

from setuptools import find_packages, setup
from itertools import chain


docs = [
    'sphinx-rtd-theme==0.5.1',
    'sphinxcontrib-youtube==1.0.1',
]

tests = {
    'atari': [
        'gym[atari,accept-rom-license]==0.23.0',
        'stable_baselines3==1.2.0',
    ],
    # Not ready: Requires Gym 0.25+
    #'avalon': [
    #    'avalon-rl==1.0.0',
    #],
    'box2d': [
        'swig==4.1.1', #unspecified box2d dep
        'gym[box2d]',
    ],
    'butterfly': [
        'pettingzoo[butterfly]',
    ],
    'crafter': [
        'crafter==1.8.0',
    ],
    'dm_control': [
        'dm_control==1.0.11',
        'gym_dmc==0.2.5',
        'gym==0.21.0',
    ],
    'dm_lab': [
        'gym_deepmindlab==0.1.2',
    ],
    #'griddly': [
    #    'imageio==2.23.0',
    #    'griddly==1.4.2',
    #],
    'magent': [
        'magent==0.2.4',
    ],
    'microrts': [
        'ffmpeg==1.4',
        'gym_microrts==0.3.2',
    ],
    'minerl': [
        'gym==0.19.0',
        'minerl==0.4.4',
    ],
    'nethack': [
        'nle==0.9.0',
    ],
    #'smac': [
    #    'git+https://github.com/oxwhirl/smac.git',
    #],
}

rllib = [
    'ray[all]==2.0.0',
    'setproctitle==1.1.10',
    'service-identity==21.1.0',
]

cleanrl = [
    'tensorboard==2.11.2',
    'torch',
    'wandb==0.13.7',
    'psutil==5.9.5',
]

flat_tests = list(set(chain.from_iterable(tests.values())))

extra_all = docs + flat_tests + rllib

setup(
    name="pufferlib",
    description="PufferAI Library"
    "PufferAI's library of RL tools and utilities",
    long_description_content_type="text/markdown",
    version=open('pufferlib/version.py').read().split()[-1].strip("'"),
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'gym==0.23',
        'numpy==1.23.3',
        'opencv-python==3.4.17.63',
        'openskill==4.0.0',
        'pettingzoo==1.19.0',
    ],
    extras_require={
        'docs': docs,
        'rllib': rllib,
        'cleanrl': cleanrl,
        'tests': flat_tests,
        'all': extra_all,
        **tests,
    },
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
