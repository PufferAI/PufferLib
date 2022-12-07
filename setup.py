from setuptools import find_packages, setup

REPO_URL = "https://github.com/PufferAI/PufferLib"

setup(
    name="pufferlib",
    description="PufferAI Library"
    "PufferAI's library of RL tools and utilities",
    long_description_content_type="text/markdown",
    version='0.1.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'ray[all]==2.0.0',
        'opencv-python==3.4.17.63',
        'nmmo==1.6.0.7',
        'openskill==2.4.0',
        'magent==0.2.4',
        'nle==0.8.1',
        #'pettingzoo[butterfly]==1.22.1'
    ],
    python_requires=">=3.7",
    license="MIT",
    author="Joseph Suarez",
    author_email="jsuarez@mit.edu",
    url=REPO_URL,
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
