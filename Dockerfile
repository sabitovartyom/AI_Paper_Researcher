# Same base image as the working LimitedMemoryRule project.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=30 \
    PIP_RETRIES=2 \
    PIP_PROGRESS_BAR=on \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libgomp1 \
        postgresql-client \
        tmux \
        vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Published CUDA 12.4 wheel; no local CMake/CUDA compilation.
# The CUDA-only index and --only-binary prevent a CPU/source fallback.
RUN python -m pip install --no-cache-dir --no-deps --only-binary=:all: \
    --index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
    llama-cpp-python==0.3.16

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
RUN python -m pip check

ARG APP_UID=1000
ARG APP_GID=1000
RUN if ! getent group "${APP_GID}" >/dev/null; then groupadd --gid "${APP_GID}" app; fi \
    && useradd --non-unique --uid "${APP_UID}" --gid "${APP_GID}" \
        --create-home --shell /bin/bash app \
    && chown -R "${APP_UID}:${APP_GID}" /app /home/app

USER app
CMD ["sleep", "infinity"]
