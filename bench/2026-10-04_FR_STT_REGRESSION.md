# Benchmark STT français Cloud9 — protocole de régression

Date de préparation : 2026-10-04.

## But

Comparer de manière reproductible les moteurs STT déjà installés dans Speaches Reloaded sur Cloud9, avec :
- WER normalisé ;
- RTF cold/warm ;
- temps mur ;
- pics GTT/VRAM ;
- même audio et même texte de référence.

Ce benchmark est une **régression synthétique**, pas un WER absolu représentatif de podcasts réels.

## Fixture

- `bench/fixtures/fr_regression_v1.txt`
- `bench/fixtures/fr_regression_v1.wav`
- voix : Piper `speaches-ai/piper-fr_FR-tom-medium`, voix `tom`
- mono, 44.1 kHz, PCM16
- durée réelle : **14.169977 s**
- le header streamable renvoyé par l'API TTS a été normalisé avec les tailles RIFF/data réelles, sans réencodage.

Texte :

> Bonjour, ceci est un test de transcription en français pour Cloud Nine. À Paris, le moteur doit reconnaître les accents, les nombres comme vingt-trois, et des noms techniques comme Vulkan, Qwen et Syncthing, sans inventer de mots.

Normalisation WER : casefold, NFKD sans diacritiques, ponctuation/apostrophes/tirets ignorés.

## Modèles installés à comparer

1. `handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf` — fr-fast
2. `handy-computer/Qwen3-ASR-1.7B-gguf` — fr-quality
3. `deepdml/faster-whisper-large-v3-turbo-ct2` — stt-whisper
4. `Systran/faster-whisper-small` — fr-small

Chaque modèle : passe cold puis warm.

## Scripts

- `scripts/cloud9_fr_stt_benchmark.py` : requêtes, WER, RTF, sampler GTT/VRAM et résultat JSON.
- `scripts/cloud9_fr_stt_benchmark_exclusive.sh` : lock `/run/cloud9-gpu.lock`, garde clients actifs, arrêt/restauration Router/Embedding/VoiceStudio/images, Speaches conservé.

Copie runtime déployée :
- `/srv/lxc/ia-compute/data/bench-tools/`
- fixture runtime : `/srv/lxc/ia-compute/data/bench-fixtures/`
- résultats runtime : `/srv/lxc/ia-compute/data/bench-results/`

## État au 2026-10-04 00:49

L'exécution de performance n'a **pas** été forcée : la tâche PBS nocturne sauvegardait successivement CT210 puis CT230 et maintenait le host entre ~70 et 75 °C avec un load élevé. La règle Cloud9 interdit de démarrer un benchmark à Tctl >=72 °C et il serait méthodologiquement mauvais de mesurer le RTF pendant un backup CPU/I/O.

Le protocole est prêt ; exécuter uniquement lorsque :
- aucune sauvegarde `proxmox-backup-client backup` n'est active ;
- Tctl <72 °C ;
- aucun job ComfyUI/AutoPublisher/LLM/VoiceStudio n'est actif ;
- le lock GPU est libre.

Commande dans CT410 :

```bash
/srv/lxc/ia-compute/data/bench-tools/cloud9_fr_stt_benchmark_exclusive.sh \
  /srv/lxc/ia-compute/data/bench-fixtures/fr_regression_v1.wav \
  /srv/lxc/ia-compute/data/bench-fixtures/fr_regression_v1.txt \
  /srv/lxc/ia-compute/data/bench-results/fr-stt-regression-v1.json
```

Ne jamais arrêter une sauvegarde PBS uniquement pour obtenir ce benchmark.
