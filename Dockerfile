# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu22.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.8

FROM ${UV_IMAGE} AS uv
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12

COPY --from=uv /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        libibverbs-dev \
        libsm6 \
        libxcb1 \
        libxext6 \
        ninja-build \
        openssh-client \
        pkg-config \
        zsh \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_TOOL_DIR=/opt/uv-tools \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    PATH=/root/.local/bin:${PATH}

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install --default ${PYTHON_VERSION} \
    && ln -sf /root/.local/bin/python${PYTHON_VERSION} /usr/bin/python${PYTHON_VERSION}

WORKDIR /opt/vllm-envs
COPY pyproject.toml README.md ./
COPY vllm_envs ./vllm_envs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv tool install --python ${PYTHON_VERSION} . \
    && mkdir -p /cache /workspaces

ENV VE_CACHE_DIR=/cache/vllm-envs \
    VE_ENVS_ROOT=/workspaces \
    UV_CACHE_DIR=/cache/uv \
    CCACHE_DIR=/cache/ccache \
    FLASHINFER_CACHE_DIR=/cache/flashinfer \
    VLLM_CACHE_ROOT=/cache/vllm \
    CCACHE_NOHASHDIR=true

VOLUME ["/cache"]
WORKDIR /workspaces

CMD ["/bin/bash"]
