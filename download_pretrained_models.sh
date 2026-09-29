#!/usr/bin/env bash
# Download the pretrained models required by infer_hdemo_codec.py:
#   1. HD-Emo codec checkpoint (model.pt) from HuggingFace (XXH333/HD-Emo)
#   2. CosyVoice2-0.5B (speech_tokenizer_v2.onnx) from ModelScope
#   3. whisper-small (ASR decoder used inside the codec) from HuggingFace
#
# All scripts anchor their local_dir to the repository root, so the
# models always land in <repo_root>/pretrained_models/.
#
# Usage:
#   ./download_pretrained_models.sh
#   PYTHON=/path/to/python ./download_pretrained_models.sh   # custom interpreter
set -e

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"

mkdir -p pretrained_models

echo "==> [1/3] Downloading HD-Emo codec checkpoint (model.pt) from HuggingFace ..."
"$PYTHON" tools/download_hd_emo.py

echo "==> [2/3] Downloading CosyVoice2-0.5B (speech_tokenizer_v2.onnx) from ModelScope ..."
"$PYTHON" tools/download_cosyvoice2.py

echo "==> [3/3] Downloading whisper-small from HuggingFace ..."
"$PYTHON" tools/download_whisper.py

echo "==> Done. All pretrained models are ready under pretrained_models/."
