<p align="center">
  <img src="docs/assets/logo.svg" width="138" alt="Logo Speaches Reloaded">
</p>

<h1 align="center">Speaches Reloaded</h1>

<p align="center">
  <strong>Une seule API vocale compatible OpenAI. CPU, Vulkan ou CUDA.</strong><br>
  Speaches + transcribe.cpp, empaquetés pour du STT/TTS auto-hébergé rapide sur NVIDIA, AMD, Intel ou CPU seul.
</p>

<p align="center">
  <img alt="Docker" src="https://img.shields.io/badge/runtime-Docker-2496ed">
  <img alt="GHCR" src="https://img.shields.io/badge/images-GHCR-54cd8a">
  <img alt="CPU Vulkan CUDA" src="https://img.shields.io/badge/backends-CPU%20%7C%20Vulkan%20%7C%20CUDA-9d8cf5">
</p>

<p align="center">🇬🇧 <a href="README.md">English</a> · 🇨🇳 <a href="README.zh-CN.md">简体中文</a></p>

---

**Speaches Reloaded** est une distribution communautaire de Speaches. Elle conserve son interface et son API compatible OpenAI, puis ajoute transcribe.cpp pour utiliser des modèles vocaux GGUF modernes sans imposer la même stack d'inférence à toutes les machines.

## ✨ Pourquoi Reloaded ?

- **3 images optimisées** — CPU, Vulkan pour AMD/Intel, CUDA pour NVIDIA.
- **Une API unique** — les clients Speaches/OpenAI gardent les mêmes endpoints.
- **Deux moteurs STT** — faster-whisper + transcribe.cpp.
- **TTS conservé** — Kokoro et Piper restent disponibles.
- **Cache modèles persistant** — une mise à jour du conteneur ne retélécharge pas les poids.
## 🖥️ Choisir son matériel

| Matériel | Image | Compose |
|---|---|---|
| CPU seul | ghcr.io/godsquantum/speaches-reloaded:latest-cpu | compose.cpu.yaml |
| AMD GPU / iGPU | ghcr.io/godsquantum/speaches-reloaded:latest-vulkan | compose.vulkan.yaml |
| Intel GPU / iGPU | ghcr.io/godsquantum/speaches-reloaded:latest-vulkan | compose.vulkan.yaml |
| NVIDIA GPU | ghcr.io/godsquantum/speaches-reloaded:latest-cuda | compose.cuda.yaml |

AMD et Intel partagent volontairement Vulkan : deux images séparées n'apporteraient aucun moteur supplémentaire.

## 🚀 Installation

~~~bash
git clone https://github.com/GodsQuantum/speaches-reloaded.git
cd speaches-reloaded
cp .env.example .env
~~~

**CPU :** docker compose -f compose.cpu.yaml up -d

**AMD / Intel :**

~~~bash
RENDER_GID="$(stat -c '%g' /dev/dri/renderD128)"
sed -i "s/^RENDER_GID=.*/RENDER_GID=$RENDER_GID/" .env
docker compose -f compose.vulkan.yaml up -d
~~~

**NVIDIA :** docker compose -f compose.cuda.yaml up -d

NVIDIA nécessite le NVIDIA Container Toolkit. L'interface et l'API sont ensuite disponibles sur le port 8000 par défaut.
## 🧠 Modèles recommandés

Après le premier démarrage :

~~~bash
./scripts/models.sh
~~~

Trois packs sont proposés :

| Pack | Contenu / usage |
|---|---|
| lean | Nemotron streaming + Kokoro + Piper |
| recommended | lean + Qwen3-ASR 0.6B + Whisper Large-v3-Turbo |
| full | recommended + Qwen3-ASR 1.7B + Parakeet v3 + diarisation Sortformer |

Le cache Hugging Face est stocké dans le volume Docker persistant speaches-models. Les poids ne sont volontairement pas inclus dans l'image.

## 🎙️ Alias pratiques

- stt-fast / fr-fast → Nemotron 3.5 ASR Streaming 0.6B
- stt-balanced → Qwen3-ASR 0.6B
- stt-quality / fr-quality → Qwen3-ASR 1.7B
- stt-whisper / whisper-1 → faster-whisper Large-v3-Turbo
- stt-europe → Parakeet TDT 0.6B v3
- tts-fast → Piper français
- tts-quality → Kokoro 82M ONNX int8

## Licence

Speaches Reloaded est une distribution communautaire non officielle. Le projet conserve l'historique Git et la licence MIT de Speaches ; transcribe.cpp reste une dépendance externe avec sa propre licence et ses notices tierces.
