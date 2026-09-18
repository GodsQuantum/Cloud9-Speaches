# Installation

Speaches Reloaded publishes three Linux x86-64 images:

| Hardware / choice | Image | Compose |
|---|---|---|
| CPU on any machine | `ghcr.io/godsquantum/speaches-reloaded:latest-cpu` | `compose.cpu.yaml` |
| AMD / Intel GPU | `ghcr.io/godsquantum/speaches-reloaded:latest-vulkan` | `compose.vulkan.yaml` |
| NVIDIA GPU | `ghcr.io/godsquantum/speaches-reloaded:latest-cuda` | `compose.cuda.yaml` |

## Common setup

~~~bash
git clone https://github.com/GodsQuantum/Speaches-Reloaded.git
cd Speaches-Reloaded
cp .env.example .env
~~~

## CPU

CPU works on any supported machine, including hosts that also have a GPU.

~~~bash
docker compose -f compose.cpu.yaml up -d
~~~

Use this profile when you want the GPU completely free.

## AMD / Intel Vulkan

~~~bash
RENDER_GID="$(stat -c '%g' /dev/dri/renderD128)"
sed -i "s/^RENDER_GID=.*/RENDER_GID=$RENDER_GID/" .env
docker compose -f compose.vulkan.yaml up -d
~~~

To keep the Vulkan image but move transcribe.cpp to CPU, set `TRANSCRIBE_BACKEND=cpu`. The CPU Compose is still preferable when you do not want the GPU device mapped at all.

## NVIDIA CUDA

Install NVIDIA Container Toolkit on the host, then:

~~~bash
docker compose -f compose.cuda.yaml up -d
~~~

Set `TRANSCRIBE_BACKEND=cpu` and `WHISPER_DEVICE=cpu` to run inference on CPU while keeping the CUDA image. Use the CPU Compose if you want Docker to stop reserving the GPU entirely.

## Models

After the container is running:

~~~bash
./scripts/models.sh
~~~

The helper offers lean, recommended and full model packs. Weights are stored in the persistent Docker model volume and are not baked into the application image.
