### Run this dockerfile from the PufferAI/puffertank repo. bash docker.sh build -d isaactank.dockerfile. bash docker.sh test
FROM pufferai/puffertank:2.0

RUN apt-get update -y \
    && apt-get install -y python3.8 \
    && apt-get install -y python3.8-dev \
    && apt-get install -y python3.8-distutils

# Install Pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 8 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 8

WORKDIR /puffertank
RUN git clone --branch pixi https://github.com/kywch/PHC
WORKDIR /puffertank/PHC
RUN python3.8 -m pip install --ignore-installed -e . 
WORKDIR /puffertank/PHC/gymtorch
RUN python3.8 -m pip install .

WORKDIR /puffertank/pufferlib
RUN git fetch --all && git checkout --track origin/isaacgym && python3.8 -m pip install -e .[cleanrl]

# Our hyperparam sweep integration
WORKDIR /puffertank/carbs
RUN git pull origin main && python3.8 -m pip install -e .

WORKDIR /puffertank
RUN echo "This container includes PHC and PufferLib. 1) Download IsaacGym from here, as we cannot distribute it: https://developer.nvidia.com/isaac-gym/download. 2) Install it from the isaacgym/python directory with python3.8 -m pip install . (NO -e). 3) Download SMPL_*.pk (Neutral, male, female) separately and copy them over. The docker folder is mounted for convenience: cp /puffertank/docker/smpl/SMPL_*.pkl resources/morph/. 4) Run training from pufferlib. python3.8 demo.py --env morph --mode train in PufferLib. 5) To eval: python3.8 demo.py --env morph --mode eval --env.headless False --env.num-envs 32. You can load models with --eval-model-path as normal in PufferLib. 6) If you need to install packages, make sure to use python3.8 -m pip install. 7) All PufferLib docs on general usage are still applicable. The default setup is for PufferLib. If you want to run PHC, install extra deps with pixi." > README.md
CMD ["/bin/bash"]
