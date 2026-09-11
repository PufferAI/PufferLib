# syntax=docker/dockerfile:1
# Build from the repository root:
# docker build --platform linux/amd64 -f ocean/fight_caves/Dockerfile -t fight-caves:local .
# The CUDA backend is compiled inside the running container; see README.md.
FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDNN_VERSION=9.13.0.50-1
ARG NCCL_VERSION=2.27.7-1+cuda13.0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN test "$(dpkg --print-architecture)" = amd64 \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        binutils build-essential ca-certificates ccache clang cmake curl git \
        libgl1-mesa-dev libgl1-mesa-dri libomp-dev libomp5 \
        libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev \
        python3-dev python3-venv x11-utils xauth xvfb \
    && curl -fsSL \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
        -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-cudart-dev-13-0 cuda-nvcc-13-0 cuda-nvml-dev-13-0 \
        cuda-nvtx-13-0 cuda-profiler-api-13-0 \
        libcublas-dev-13-0 libcurand-dev-13-0 libcusolver-dev-13-0 \
        "libcudnn9-cuda-13=${CUDNN_VERSION}" \
        "libcudnn9-dev-cuda-13=${CUDNN_VERSION}" \
        "libcudnn9-headers-cuda-13=${CUDNN_VERSION}" \
        "libnccl2=${NCCL_VERSION}" "libnccl-dev=${NCCL_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda-13.0 \
    VIRTUAL_ENV=/workspace/PufferLib/.venv \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
ENV PATH="${VIRTUAL_ENV}/bin:${CUDA_HOME}/bin:${PATH}"

# The host driver is injected by NVIDIA Container Toolkit at runtime. Puffer's
# -lnvidia-ml also needs this unversioned linker name; it is dangling at build time.
RUN ln -s libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so \
    && printf '%s\n' "${CUDA_HOME}/lib64" > /etc/ld.so.conf.d/fight-caves-cuda.conf \
    && ldconfig \
    && printf '%s\n' 'export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"' \
        > /etc/profile.d/fight-caves-env.sh

WORKDIR /workspace/PufferLib
COPY ocean/fight_caves/docker-constraints.txt /opt/fight-caves/constraints.txt
RUN python3 -m venv --prompt fight-caves "$VIRTUAL_ENV" \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        -c /opt/fight-caves/constraints.txt \
        --index-url https://download.pytorch.org/whl/cu130 torch

# Dockerfile.dockerignore limits this context to Puffer and Fight Caves sources.
COPY . .
RUN python -m pip install --no-cache-dir -c /opt/fight-caves/constraints.txt -e . pytest \
    && python -m pip check \
    && ./build.sh fight_caves --fast \
    && bash tests/fight_caves.sh test --core

CMD ["bash"]
