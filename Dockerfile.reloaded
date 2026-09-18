# syntax=docker/dockerfile:1.7
ARG TRANSCRIBE_BUILDER_IMAGE=ubuntu:24.04
ARG BASE_IMAGE=ubuntu:24.04
FROM ${TRANSCRIBE_BUILDER_IMAGE} AS transcribe-builder

ARG TRANSCRIBE_REF=be7a8b35e9ba2df20298bd26e32d53407c3bcbcd
ARG TRANSCRIBE_BACKEND=cpu

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake git glslc libopenblas-dev libvulkan-dev ninja-build spirv-headers \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --filter=blob:none https://github.com/handy-computer/transcribe.cpp.git \
 && cd transcribe.cpp \
 && git checkout "${TRANSCRIBE_REF}"

RUN case "${TRANSCRIBE_BACKEND}" in \
      cpu) BACKEND_FLAGS="" ;; \
      vulkan) BACKEND_FLAGS="-DTRANSCRIBE_VULKAN=ON" ;; \
      cuda) command -v nvcc >/dev/null; BACKEND_FLAGS="-DTRANSCRIBE_CUDA=ON" ;; \
      *) echo "Unsupported TRANSCRIBE_BACKEND=${TRANSCRIBE_BACKEND}" >&2; exit 2 ;; \
    esac \
 && cmake -S transcribe.cpp -B transcribe.cpp/build -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DTRANSCRIBE_BUILD_SHARED=ON \
      -DTRANSCRIBE_BUILD_TESTS=OFF \
      -DTRANSCRIBE_BUILD_EXAMPLES=OFF \
      -DTRANSCRIBE_BUILD_TOOLS=OFF \
      -DTRANSCRIBE_USE_SYSTEM_BLAS=ON \
      ${BACKEND_FLAGS} \
 && cmake --build transcribe.cpp/build --parallel \
 && cmake --install transcribe.cpp/build --prefix /opt/transcribe
RUN mkdir -p /opt/transcribe/python \
 && cp -a transcribe.cpp/bindings/python/src/transcribe_cpp /opt/transcribe/python/

FROM ${BASE_IMAGE} AS runtime

ARG TRANSCRIBE_BACKEND=cpu
LABEL org.opencontainers.image.title="Speaches Reloaded" \
      org.opencontainers.image.description="Speaches with transcribe.cpp CPU, Vulkan and CUDA backends" \
      org.opencontainers.image.source="https://github.com/GodsQuantum/speaches-reloaded" \
      org.opencontainers.image.licenses="MIT"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl ffmpeg libopenblas0-pthread \
 && if [ "${TRANSCRIBE_BACKEND}" = "vulkan" ]; then \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libvulkan1 mesa-vulkan-drivers; \
    fi

RUN useradd --create-home --shell /bin/bash --uid 1000 ubuntu || true
USER ubuntu
ENV HOME=/home/ubuntu \
    PATH=/home/ubuntu/.local/bin:$PATH \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/home/ubuntu/.cache/uv \
    UV_PYTHON_CACHE_DIR=/home/ubuntu/.cache/uv/python
WORKDIR $HOME/speaches

COPY --chown=ubuntu --from=ghcr.io/astral-sh/uv:0.10 /uv /bin/uv
RUN --mount=type=cache,target=/home/ubuntu/.cache/uv,uid=1000,gid=1000 \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --compile-bytecode --no-install-project --no-dev
COPY --chown=ubuntu . .
RUN --mount=type=cache,target=/home/ubuntu/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --compile-bytecode --no-dev
RUN mkdir -p $HOME/.cache/huggingface/hub

USER root
COPY --from=transcribe-builder /opt/transcribe/ /opt/transcribe/
RUN find /home/ubuntu/speaches/.venv -maxdepth 7 -path "*/nvidia/*/lib" -type d \
      > /etc/ld.so.conf.d/venv-nvidia.conf \
 && echo "/opt/transcribe/lib" > /etc/ld.so.conf.d/transcribe.conf \
 && ldconfig
USER ubuntu

ENV UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000 \
    PATH="/home/ubuntu/speaches/.venv/bin:$PATH" \
    PYTHONPATH="/opt/transcribe/python:/home/ubuntu/speaches/src" \
    TRANSCRIBE_LIBRARY="/opt/transcribe/lib/libtranscribe.so" \
    TRANSCRIBE_BACKEND="${TRANSCRIBE_BACKEND}" \
    LD_LIBRARY_PATH="/opt/transcribe/lib" \
    DO_NOT_TRACK=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYANNOTE_METRICS_ENABLED=0

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1
CMD ["uvicorn", "--factory", "speaches.main:create_app"]
