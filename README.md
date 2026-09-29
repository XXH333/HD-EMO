# HD-Emo

HD-Emo is an emotional speech codec. Given an utterance, it extracts:

- **frame-level speech tokens** and two kinds of **frame-level preference tokens** (emotion / content),
- **word-level VAD** (Valence-Arousal-Dominance),**
- sentence-level emotion recognition** (9 classes),
- **ASR transcription** of the utterance.

## 1. Python environment

```bash
conda create -n hdemo python=3.10 -y
conda activate hdemo
```

## 2. Clone the repository

```bash
git clone https://github.com/XXH333/HD-EMO.git
cd HD-EMO
```

## 3. Install dependencies

```bash
pip install -e .
```

> On Linux the PyPI `torch` wheels ship with CUDA support. To match a
> specific CUDA version, install torch first, e.g.
> `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121`,
> then run `pip install -e .`.
>
## 4. Download pretrained models

```bash
./download_pretrained_models.sh
```

This downloads everything into `pretrained_models/`:

| Path | Source | Description |
|---|---|---|
| `HD-Emo/model.pt` | HuggingFace `XXH333/HD-Emo` | HD-Emo codec checkpoint |
| `CosyVoice2-0.5B/speech_tokenizer_v2.onnx` | ModelScope `iic/CosyVoice2-0.5B` | speech tokenizer (ONNX) |
| `whisper/` | HuggingFace `openai/whisper-small` | ASR decoder used inside the codec |

If the HuggingFace repos are gated or private, log in first with
`huggingface-cli login` (or export `HF_TOKEN`).

## Run

```bash
python infer_hdemo_codec.py
```
