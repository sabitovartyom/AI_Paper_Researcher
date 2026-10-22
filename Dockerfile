# syntax=docker/dockerfile:1

# CUDA 12.x is required for Pascal GPUs. GTX 1080 Ti has compute capability 6.1.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        less \
        libgomp1 \
        pkg-config \
        postgresql-client \
        tmux \
        vim \
    && rm -rf /var/lib/apt/lists/*

# Reuse the base image's CUDA-enabled PyTorch, while keeping project packages
# writable by the development user. A bind mount at /app cannot hide this venv.
RUN python -m venv --system-site-packages /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
        pip==25.3 \
        wheel==0.45.1 \
        setuptools==80.9.0 \
        cmake==3.31.6 \
        ninja==1.11.1.4

# Compile explicitly: an ordinary wheel may contain only the CPU backend.
# This layer is cached even when the application dependency list changes.
ARG CUDA_ARCHITECTURES=61
ARG BUILD_JOBS=4
RUN CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
    CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" \
    python -m pip install -v --no-cache-dir --no-binary=llama-cpp-python llama-cpp-python==0.3.36

COPY requirements.txt /opt/requirements/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r /opt/requirements/requirements.txt \
    && python -m pip check

ARG APP_UID=1000
ARG APP_GID=1000
RUN if ! getent group "${APP_GID}" >/dev/null; then groupadd --gid "${APP_GID}" app; fi \
    && useradd --non-unique --uid "${APP_UID}" --gid "${APP_GID}" \
        --create-home --shell /bin/bash app \
    && mkdir -p /app \
    && chown -R "${APP_UID}:${APP_GID}" /app /home/app /opt/venv

USER app
WORKDIR /app

# Source code and model weights are supplied by launch_container's bind mount.
# Keep the development container running between docker exec sessions.
CMD ["sleep", "infinity"]
