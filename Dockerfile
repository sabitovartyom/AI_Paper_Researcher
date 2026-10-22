# syntax=docker/dockerfile:1

# CUDA 12.x is required for Pascal GPUs. GTX 1080 Ti has compute capability 6.1.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=30 \
    PIP_RETRIES=2 \
    PIP_PROGRESS_BAR=on \
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
        scikit-build-core==0.11.6 \
        cmake==3.31.6 \
        ninja==1.11.1.4

# Download separately with a byte/speed meter and bounded network waits.
# Pin the immutable PyPI source URL and verify its published SHA-256.
RUN mkdir -p /opt/llama-source \
    && curl --fail --location --show-error --progress-meter --ipv4 \
        --connect-timeout 15 \
        --max-time 300 \
        --speed-limit 1024 \
        --speed-time 30 \
        --output /opt/llama-source/llama_cpp_python-0.3.36.tar.gz \
        https://files.pythonhosted.org/packages/ec/e9/e7de2b0463ea3ffbf0ede6cb21b58c1258a8f6521aae45ca773a59fe7cf3/llama_cpp_python-0.3.36.tar.gz \
    && printf '%s  %s\n' \
        832db0699007f1be95a7e41ef12e88926b02ba836461e36a36372db2760c1a2e \
        /opt/llama-source/llama_cpp_python-0.3.36.tar.gz \
        | sha256sum --check -

# Compile explicitly: an ordinary wheel may contain only the CPU backend.
# This pip invocation uses the local source archive, with package downloads
# disabled. Runtime dependencies are installed from requirements.txt below.
ARG CUDA_ARCHITECTURES=61
ARG BUILD_JOBS=4
RUN CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
    CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" \
    python -m pip install -v \
        --no-cache-dir \
        --no-index \
        --no-deps \
        --no-build-isolation \
        /opt/llama-source/llama_cpp_python-0.3.36.tar.gz

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
