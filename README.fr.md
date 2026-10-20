<p align="center">
  <img src="https://raw.githubusercontent.com/GodsQuantum/Cloud9-Speaches/main/docs/assets/speaches-reloaded-logo.svg" width="138" alt="Logo Cloud9-Speaches">
</p>

<h1 align="center">Cloud9-Speaches</h1>

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

**Cloud9-Speaches** est une distribution communautaire de Speaches. Elle conserve son interface et son API compatible OpenAI, puis ajoute transcribe.cpp pour utiliser des modèles vocaux GGUF modernes sans imposer la même stack d'inférence à toutes les machines.

## ✨ Pourquoi Cloud9-Speaches ?

- **3 images optimisées** — CPU, Vulkan pour AMD/Intel, CUDA pour NVIDIA.
- **Une API unique** — les clients Speaches/OpenAI gardent les mêmes endpoints.
- **Trois exécutants STT** — transcribe.cpp pour les GGUF Nemotron/Qwen3-ASR, faster-whisper pour Whisper Turbo CT2 et Parakeet ONNX en option.
- **TTS conservé** — Kokoro et Piper restent disponibles.
- **Cache modèles persistant** — une mise à jour du conteneur ne retélécharge pas les poids.

## 🎯 Sous-titres mot à mot et benchmarks Radeon

Les modèles ne fournissent pas tous des timestamps natifs fiables : Qwen3-ASR nécessite un alignement forcé ; Nemotron peut émettre un mot vide et dépasser la durée audio. Whisper Turbo fournit des temps par mot qu'il faut également contrôler. Le pont **optionnel** vers le service WhisperX AutoSubs déjà installé vérifie les mots et les intervalles. Éviter un double alignement quand AutoSubs fait déjà le sien. Voir [contrat word-for-word](docs/cloud9-autosubs-word-timestamps-20261013.md).

Les mesures CPU sur Ryzen 7 8845HS, les versions/quantifications et le protocole **CPU contre Vulkan Radeon 780M** sont dans le [dossier de benchmarks AMD](docs/cloud9-amd-780m-asr-cpu-vulkan-benchmarks-20261013.md). Les gains GPU ne sont pas revendiqués sans un GPU sain et des tests réels. Le moteur faster-whisper/CT2 n'acquiert pas le support Vulkan AMD simplement en configurant transcribe.cpp.

Les images GHCR ne sont plus publiées automatiquement sur chaque fusion de `main` : un tag de release ou une action manuelle est nécessaire.

## 🖥️ Choisir son matériel

| Matériel | Image | Compose |
|---|---|---|
| CPU seul | ghcr.io/godsquantum/speaches-reloaded:latest-cpu | compose.cpu.yaml |
| AMD GPU / iGPU | ghcr.io/godsquantum/speaches-reloaded:latest-vulkan | compose.vulkan.yaml |
| Intel GPU / iGPU | ghcr.io/godsquantum/speaches-reloaded:latest-vulkan | compose.vulkan.yaml |
| NVIDIA GPU | ghcr.io/godsquantum/speaches-reloaded:latest-cuda | compose.cuda.yaml |

AMD et Intel partagent volontairement Vulkan : deux images séparées n'apporteraient aucun moteur supplémentaire.

**Le CPU reste toujours utilisable, même si la machine possède un GPU.** Utilise `compose.cpu.yaml` sur n'importe quelle machine x86-64 si tu veux libérer complètement le GPU. Avec l'image Vulkan déjà déployée, `TRANSCRIBE_BACKEND=cpu` bascule transcribe.cpp sur le CPU. Avec l'image CUDA, mets aussi `WHISPER_DEVICE=cpu`. Pour que Docker ne réserve même plus le GPU, le profil CPU reste le choix le plus propre.

## 🚀 Installation

~~~bash
git clone https://github.com/GodsQuantum/Cloud9-Speaches.git
cd Cloud9-Speaches
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
bash ./scripts/models.sh
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

Cloud9-Speaches est une distribution communautaire non officielle. Le projet conserve l'historique Git et la licence MIT de Speaches ; transcribe.cpp reste une dépendance externe avec sa propre licence et ses notices tierces.

## Conseil automatique de profil CPU, Vulkan ou CUDA

**Téléchargements :** [candidats natifs cinq architectures](https://github.com/GodsQuantum/Cloud9-Speaches/releases/tag/desktop-v0.1.0-rc.1). Ils ont passé un vrai test acoustique CPU, mais **pas la signature officielle Windows, la notarisation Apple Developer ID ni la validation GPU sur toutes les machines**.

Le diagnostic est prévu sur **l'hôte Linux Docker** et ne modifie pas le système. Il conseille CUDA si nvidia-smi fonctionne, Vulkan si le render node est accessible ET si vulkaninfo réussit, sinon CPU :

~~~bash
python3 scripts/auto-profile.py
python3 scripts/auto-profile.py --env
docker compose -f compose.vulkan.yaml config    # examiner d'abord la recommandation
~~~

Sur Ryzen 7 8845HS / Radeon 780M, tester transcribe.cpp GGUF en Vulkan ; faster-whisper/CTranslate2 reste en CPU avec WHISPER_DEVICE=cpu (environ 4 threads). CUDA exige NVIDIA Container Toolkit et une inférence réelle. Sur petite machine, commencer par compose.cpu.yaml et les modèles lean. Mesurer Nemotron, Qwen3-ASR, Whisper, FastConformer FR Q4 sur le même audio : mémoire, latence, précision et temps par mot. **Détecter un GPU ne certifie pas un gain de vitesse.**
