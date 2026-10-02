#!/usr/bin/env python3

import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import numpy as np
import onnxruntime
import torch
import torchaudio
import whisper
from hyperpyyaml import load_hyperpyyaml

def load_audio(path):
    audio, sample_rate = torchaudio.load(path, backend='soundfile')
    if audio.size(0) > 1:
        audio = audio[:1]
    if sample_rate != 16000:
        audio = torchaudio.functional.resample(audio, sample_rate, 16000)

    max_val = audio.abs().max()
    if max_val > 1:
        audio = audio / max_val

    if audio.size(1) > 30 * 16000:
        raise ValueError('Audio longer than 30 seconds is not supported')

    return audio

def extract_speech_token(audio, ort_session):
    feat = whisper.log_mel_spectrogram(audio, n_mels=128)
    inputs = {
        ort_session.get_inputs()[0].name: feat.cpu().numpy(),
        ort_session.get_inputs()[1].name: np.array(
            [feat.shape[2]], dtype=np.int32
        )
    }

    speech_token = ort_session.run(None, inputs)[0].reshape(-1)

    if speech_token.min() < 0 or speech_token.max() >= 6561:
        raise ValueError(
            f'Invalid speech token range: '
            f'[{speech_token.min()}, {speech_token.max()}]'
        )

    return torch.from_numpy(
        speech_token.astype(np.int64)
    ).unsqueeze(0)


def load_codec(config_path, checkpoint_path, whisper_path, device):
    with open(config_path, 'r') as f:
        configs = load_hyperpyyaml(
            f,
            overrides={'whisper_path': whisper_path}
        )

    model = configs['hdemo_codec']
    state_dict = torch.load(
        checkpoint_path,
        map_location=device
    )

    # Strip training metadata and strictly validate weights
    # against the model structure
    state_dict.pop('epoch', None)
    state_dict.pop('step', None)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    return model


if __name__ == '__main__':
    # ==================== Configuration ====================
    audio_path = os.path.join(
        project_root,
        'demo/demo.mp3'
    )
    # Word-level alignment info (JSON literal, used for word-level VAD extraction)
    aligned_info = {"aligned_words": [{"word": "just", "start": 0.3, "end": 0.55}, {"word": "where", "start": 0.56, "end": 0.73}, {"word": "the", "start": 0.73, "end": 0.82}, {"word": "cliff", "start": 0.85, "end": 1.08}, {"word": "goes", "start": 1.13, "end": 1.36}, {"word": "down", "start": 1.39, "end": 1.62}, {"word": "a", "start": 1.62, "end": 1.72}, {"word": "hundred", "start": 1.72, "end": 1.99}, {"word": "fathoms", "start": 2.07, "end": 2.46}, {"word": "sheer", "start": 2.46, "end": 2.99}, {"word": "a", "start": 3.31, "end": 3.37}, {"word": "wall", "start": 3.37, "end": 3.68}, {"word": "of", "start": 3.68, "end": 3.85}, {"word": "rock", "start": 3.85, "end": 4.18}, {"word": "to", "start": 4.34, "end": 4.48}, {"word": "where", "start": 4.48, "end": 4.67}, {"word": "the", "start": 4.67, "end": 4.82}, {"word": "river", "start": 4.82, "end": 5.17}, {"word": "foams", "start": 5.26, "end": 5.57}, {"word": "along", "start": 5.57, "end": 5.89}, {"word": "its", "start": 5.89, "end": 6.08}, {"word": "bed", "start": 6.14, "end": 6.47}, {"word": "i've", "start": 7.12, "end": 7.27}, {"word": "often", "start": 7.27, "end": 7.67}, {"word": "wondered", "start": 7.67, "end": 8.17}, {"word": "who", "start": 8.24, "end": 8.37}, {"word": "was", "start": 8.37, "end": 8.56}, {"word": "brave", "start": 8.64, "end": 9.02}, {"word": "to", "start": 9.11, "end": 9.2}, {"word": "plant", "start": 9.22, "end": 9.5}, {"word": "a", "start": 9.5, "end": 9.6}, {"word": "cross", "start": 9.63, "end": 10.0}, {"word": "on", "start": 10.0, "end": 10.2}, {"word": "such", "start": 10.2, "end": 10.47}, {"word": "an", "start": 10.47, "end": 10.6}, {"word": "edge", "start": 10.6, "end": 11.12}]}
    aligned_words = aligned_info['aligned_words']

    gt_text = "just where the cliff goes down a hundred fathoms sheer a wall of rock to where the river foams along its bed i've often wondered who was brave to plant a cross on such an edge"

    config_path = os.path.join(
        project_root,
        'conf/hdemo_codec.yaml'
    )

    checkpoint_path = os.path.join(
        project_root,
        'pretrained_models/HD-Emo/model.pt'
    )

    whisper_path = os.path.join(
        project_root,
        'pretrained_models/whisper'
    )

    onnx_path = os.path.join(
        project_root,
        'pretrained_models/CosyVoice2-0.5B/'
        'speech_tokenizer_v2.onnx'
    )

    language = 'en'
    max_new_tokens = 256
    device = torch.device('cuda')

    # Must match the emotion label index order used in training
    emotion_labels = [
        'Angry',      # 0: angry
        'Disgusted',  # 1: disgusted
        'Fearful',    # 2: fearful
        'Happy',      # 3: happy
        'Neutral',    # 4: neutral
        'Other',      # 5: other
        'Sad',        # 6: sad
        'Surprised',  # 7: surprised
        'Unknown'     # 8: unknown
    ]

    vad_labels = [
        'valence',
        'arousal',
        'dominance'
    ]

    emotion_code_labels = [
        'emotion_fsq_dim_0',
        'emotion_fsq_dim_1',
        'emotion_fsq_dim_2'
    ]

    content_code_labels = [
        'content_fsq_dim_0',
        'content_fsq_dim_1',
        'content_fsq_dim_2',
        'content_fsq_dim_3'
    ]

    # ==================== ONNX Tokenizer ====================

    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = 1

    available_providers = onnxruntime.get_available_providers()

    if 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    print(f'ONNX provider: {providers[0]}')

    ort_session = onnxruntime.InferenceSession(
        onnx_path,
        sess_options=options,
        providers=providers
    )

    # ==================== Speech Token Extraction ====================

    audio = load_audio(audio_path)
    speech_token = extract_speech_token(
        audio,
        ort_session
    )

    speech_token_len = torch.tensor(
        [speech_token.size(1)],
        dtype=torch.long
    )

    print(f'audio duration: {audio.size(1) / 16000:.3f}s')
    print(f'speech token shape: {tuple(speech_token.shape)}')

    # ==================== Codec Loading ====================

    model = load_codec(
        config_path,
        checkpoint_path,
        whisper_path,
        device
    )

    # ==================== Codec Inference ====================

    with torch.inference_mode():
        result = model.inference(
            {
                'speech_token': speech_token,
                'speech_token_len': speech_token_len
            },
            device=device,
            language=language,
            max_new_tokens=max_new_tokens,
            align_words=aligned_words
        )

    # ==================== Collect Results ====================

    emotion_class = result['emotion_class'][0].item()
    emotion_probability = (
        result['emotion_probability'][0].tolist()
    )

    emotion_token = result['emotion_token'][0].tolist()
    content_token = result['content_token'][0].tolist()
    emotion_code = result['emotion_code'][0].tolist()
    content_code = result['content_code'][0].tolist()
    frame_vad = result['frame_vad'][0].tolist()
    asr_text = result['asr_text'][0].strip()

    word_vad = result['word_vad']
    word_vad_index = result['word_vad_index']

    emotion_probability_labeled = {
        label: probability
        for label, probability in zip(
            emotion_labels,
            emotion_probability
        )
    }

    frame_vad_labeled = [
        {
            'frame': i,
            'start_time': i / model.token_frame_rate,
            'end_time': (i + 1) / model.token_frame_rate,
            'valence': vad[0],
            'arousal': vad[1],
            'dominance': vad[2]
        }
        for i, vad in enumerate(frame_vad)
    ]

    # Word-level VAD: keep only words whose timestamp range maps to
    # valid frames (consistent with training; words with empty
    # frame ranges are skipped)
    if word_vad is not None:
        word_vad_labeled = [
            {
                'word': aligned_words[idx]['word'],
                'start_time': aligned_words[idx]['start'],
                'end_time': aligned_words[idx]['end'],
                'valence': vad[0],
                'arousal': vad[1],
                'dominance': vad[2]
            }
            for idx, vad in zip(
                word_vad_index.tolist(),
                word_vad.tolist()
            )
        ]
    else:
        word_vad_labeled = []

    output = {
        'audio': audio_path,
        'audio_duration': audio.size(1) / 16000,
        'gt_text': gt_text,
        'asr_text': asr_text,

        'frame_rate': model.token_frame_rate,
        'num_frames': len(emotion_token),

        'emotion_class': {
            'index': emotion_class,
            'label': emotion_labels[emotion_class],
            'probability': emotion_probability[emotion_class]
        },

        'emotion_class_labels': {
            str(index): label
            for index, label in enumerate(emotion_labels)
        },

        'emotion_probability': emotion_probability,
        'emotion_probability_labeled': (
            emotion_probability_labeled
        ),

        'emotion_token': emotion_token,
        'content_token': content_token,

        'emotion_code': emotion_code,
        'content_code': content_code,

        'vad_dimension_order': vad_labels,
        'vad_dimension_labels': {
            'valence': (
                'Valence: higher values indicate more positive emotion'
            ),
            'arousal': (
                'Arousal: higher values indicate more intense emotion'
            ),
            'dominance': (
                'Dominance: higher values indicate more assertive emotion'
            )
        },

        'frame_vad': frame_vad,
        'frame_vad_labeled': frame_vad_labeled,

        'word_vad': word_vad_labeled
    }

    # ==================== Print Results ====================

    print()
    print('========== Inference Result ==========')
    print(f'gt text: {gt_text}')
    print(f'asr text: {asr_text}')

    print(f'style token: {emotion_token}')
    print(f'content token: {content_token}')
    
    print(
        f'emotion class: {emotion_class} '
        f'({emotion_labels[emotion_class]})'
    )
    print(
        f'emotion probability: '
        f'{emotion_probability[emotion_class]:.6f}'
    )

    print('all emotion probabilities:')
    for label, probability in (
        emotion_probability_labeled.items()
    ):
        print(f'  {label:<10}: {probability:.6f}')

    print()
    if len(word_vad_labeled) > 0:
        print(
            f'word-level VAD: '
            f'{len(word_vad_labeled)} words'
        )
        print(
            f'  {"word":<12} {"start":>7} {"end":>7}'
            f' {"Valence":>9} {"Arousal":>9} {"Dominance":>10}'
        )
        for word_vad_info in word_vad_labeled:
            print(
                f'  {word_vad_info["word"]:<12} '
                f'{word_vad_info["start_time"]:>7.3f} '
                f'{word_vad_info["end_time"]:>7.3f} '
                f'{word_vad_info["valence"]:>9.6f} '
                f'{word_vad_info["arousal"]:>9.6f} '
                f'{word_vad_info["dominance"]:>10.6f}'
            )
    else:
        print('word-level VAD: no valid aligned words')

    print()
    print(f'frame number: {output["num_frames"]}')