import random
from typing import Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F

from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig
from transformers.modeling_outputs import BaseModelOutput
from jiwer import wer

from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.utils.mask import make_pad_mask
from cosyvoice.utils.losses import CCCLoss
from copy import deepcopy

class WhisperEncoder(torch.nn.Module):
    def __init__(self, whisper_path='./pretrained_models/whisper'):
        super().__init__()
        
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_path)
        self.whisper_config = WhisperConfig.from_pretrained(whisper_path)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_path)

        self.whisper_decoder = self.whisper_model.get_decoder()
        # for param in self.whisper_decoder.parameters():
        #     param.requires_grad = False

        self.proj_out = self.whisper_model.proj_out
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, raw_text: list, token_embedded_for_asr: torch.Tensor, token_len: torch.Tensor = None):
        device = token_embedded_for_asr.device

        # input_ids here serve as the labels for the ASR task
        # process the whole batch at once
        input_labels = self.whisper_processor.tokenizer(
            text=raw_text, # pass the list of raw texts
            return_tensors="pt",
            padding="longest", # pad to the longest text in the batch
            truncation=True # truncate overly long texts
        ).input_ids.to(device)

        pad_id = self.whisper_processor.tokenizer.pad_token_id
        fill_values = torch.full((len(raw_text), 1), pad_id).to(device)
        # shift left by one (drop the first token), then pad on the right
        target_labels = torch.cat([input_labels[:, 1:], fill_values], dim=1)

        decoder_outputs = self.whisper_decoder(
            input_ids=input_labels,
            encoder_hidden_states=token_embedded_for_asr,
            # cross_attn_mask = attn_mask
        )
        logits = decoder_outputs[0]
        del decoder_outputs

        logits = self.proj_out(logits)
        whisper_loss = self.criterion(logits.view(-1, logits.size(-1)), target_labels.view(-1))

        predicted_labels = torch.argmax(logits, dim=-1)
        predicted_text = self.whisper_processor.batch_decode(predicted_labels, skip_special_tokens=True)
        error_rate = torch.tensor(wer(raw_text, predicted_text))

        return whisper_loss, error_rate, raw_text, predicted_text

    @torch.inference_mode()
    def transcribe(self, encoder_hidden_states, encoder_attention_mask=None, language='zh', max_new_tokens=256):
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(language=language, task='transcribe')
        token_ids = self.whisper_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=max_new_tokens
        )
        return self.whisper_processor.batch_decode(token_ids, skip_special_tokens=True)


class AdaptiveSpeechTokenCodec(torch.nn.Module):
    def __init__(
            self,
            hidden_size: 512,
            speech_token_size: 6561,
            emo_fsq_dim: 4,
            emo_fsq_scale: 5,
            con_fsq_dim: 4,
            con_fsq_scale: 8,
            token_frame_rate: 25,
            spk: False,
            # emotion_encoder: torch.nn.Module,
            whisper_encoder: torch.nn.Module,
            decoupler: torch.nn.Module,
            adapter: torch.nn.Module,
            combiner: torch.nn.Module,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.token_frame_rate = token_frame_rate
        self.speech_token_size = speech_token_size
        self.emotion_token_size = emo_fsq_scale**emo_fsq_dim
        self.content_token_size = con_fsq_scale**con_fsq_dim # 2401
        self.emotion_fsq_dim = emo_fsq_dim
        self.emotion_fsq_scale = emo_fsq_scale
        self.content_fsq_dim = con_fsq_dim
        self.content_fsq_scale = con_fsq_scale
        self.spk = spk
        if spk is True:
            self.spk_embed_affine_layer = torch.nn.Linear(512, hidden_size)

        self.speech_token_embedding = torch.nn.Embedding(speech_token_size, hidden_size) # hidden_size//2) # exit()

        # emotion preference tokens
        self.emotion_decoupler = decoupler
        self.emotion_adapter = adapter
        self.emotion_encoder = deepcopy(adapter)
        # sentence-level emotion label
        self.emotion_query_embedding = torch.nn.Embedding(9, hidden_size)
        self.emotion_score_encoder = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=16,
            batch_first=True
        )
        self.global_emotion_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.SiLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.SiLU(),
        )
        self.global_emo_criterion = nn.CrossEntropyLoss() # supports soft labels
        # word-level emotion branch
        self.word_pre_pool_net = nn.Sequential( # input: [B, L, hidden_size] -> [Total_Words, hidden_size]
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        self.word_post_pool_net = nn.Sequential( # [Total_Words, hidden_size]
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 3), # output: V, A, D
            nn.Sigmoid() # assumes VAD targets are normalized to [0, 1]
        )

        # content preference tokens
        self.content_decoupler = deepcopy(decoupler)
        self.content_adapter = deepcopy(adapter)
        self.whisper_input_dim = whisper_encoder.whisper_config.d_model # 768
        self.whisper_affine_layer = torch.nn.Linear(hidden_size, self.whisper_input_dim)
        self.content_token_encoder = whisper_encoder
        # freeze
        modules_to_freeze = [
            self.content_decoupler,
            self.content_adapter,
            self.whisper_affine_layer,
            self.content_token_encoder
        ]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
            module.eval()

        # combiner
        self.modulation_layer = nn.Sequential( # emotion
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size) 
        )
        self.modulation_adapter = deepcopy(adapter) # content
        self.combiner = combiner

        self.emotion_FSQ_affine_linear = torch.nn.Linear(hidden_size, emo_fsq_dim)
        self.content_FSQ_affine_linear = torch.nn.Linear(hidden_size, con_fsq_dim)
        self.emotion_token_affine_linear = torch.nn.Linear(emo_fsq_dim, hidden_size)
        self.content_token_affine_linear = torch.nn.Linear(con_fsq_dim, hidden_size)

        self.output_affine_linear = torch.nn.Linear(hidden_size, speech_token_size)
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_ID)
        self.mse_loss = nn.MSELoss() 
        
    def FSQ(self, codes, ty='content'):
        # FSQ quantization: [B, L, dim] -> [B, L]
        # values: [0, 1] -> [0, codebook)
        if ty=='content':
            scale = self.content_fsq_scale
        else:
            scale = self.emotion_fsq_scale
        # 1. round to the nearest code
        codes = torch.clamp(codes, min=0., max=1.) # guard against rounding beyond the max when noise is large
        scaled_codes = codes * (scale - 1)
        rounded_codes = torch.round(scaled_codes)
        quantized_codes = rounded_codes / (scale - 1)

        # 2. compute token indices
        splits = torch.split(rounded_codes, 1, dim=2)
        indexs = torch.zeros_like(rounded_codes)[:, :, 0:1]

        for i, split in enumerate(splits):
            indexs = indexs + split * scale**i
            # indexs = indexs + split * scale**(len(splits) - 1 - i)

        # 3. apply the straight-through estimator (STE)
        codes_out = codes + (quantized_codes - codes).detach()

        return codes_out, indexs.squeeze(-1).to(torch.int32)

    def aFSQ(self, indexs, ty='content'):
        # inverse FSQ: [B, L] -> [B, L, dim]
        # values: [0, codebook) -> [0, 1]
        if ty=='content':
            scale = self.content_fsq_scale
            dim = self.content_fsq_dim
        else:
            scale = self.emotion_fsq_scale
            dim = self.emotion_fsq_dim
        hidden_list = []
        h = torch.zeros_like(indexs).to(indexs.device)
        for i in reversed(range(dim)):
            indexs = indexs - h
            h = indexs // scale**i
            hidden_list.insert(0, h)
            # hidden_list.append(h)
            h = h * scale**i
        codes = torch.stack(hidden_list, dim=-1) / (scale - 1)

        return codes.to(indexs.device)

    def accelerate(self, pre_token, target_token):
        mask = target_token != IGNORE_ID
        # count matching elements
        similar_elements = (pre_token == target_token) & mask
        similar_count = similar_elements.sum().item()
        # count total valid elements
        total_valid_elements = mask.sum().item()
        # compute accuracy
        acc = similar_count / total_valid_elements if total_valid_elements > 0 else 0

        return torch.tensor(acc)

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            speech_token: (B, L)
            speech_token_len: (B)
            emotion: (B, 9)
            content: (B, L)
            align_words: list, (B, L2, 5), [start_time, end_time, valence, arousal, dominance]
        """
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        emotion = batch['emotion'].to(device)
        content = batch['text']
        align_words = batch['align_words']

        # 1. prepare target: target_token for accuracy comparison,
        # true_feature for the loss
        target_token = speech_token.clone()
        token_mask = torch.arange(target_token.size(1)).to(device) >= speech_token_len.unsqueeze(1)
        target_token = target_token.masked_fill(token_mask, IGNORE_ID)

        # 2. decouple speech_token
        speech_token = self.speech_token_embedding(speech_token)
        emotion_token, emotion_token_mask = self.emotion_decoupler(speech_token, speech_token_len) # [B, L, 512]
        content_token, content_token_mask = self.content_decoupler(speech_token, speech_token_len) # [B, L, 512]

        emotion_token = emotion_token.masked_fill(~emotion_token_mask.transpose(1, 2), 0)
        content_token = content_token.masked_fill(~content_token_mask.transpose(1, 2), 0)

        # project down for reconstruction
        emotion_token_gn = torch.sigmoid(self.emotion_FSQ_affine_linear(emotion_token))
        content_token_gn, _ = self.modulation_adapter(content_token.detach(), speech_token_len) # refine the acoustic skeleton
        content_token_gn = torch.sigmoid(self.content_FSQ_affine_linear(content_token_gn))

        # 3. emotion token: hierarchical supervision
        # alignment
        emotion_token, emotion_mask = self.emotion_adapter(emotion_token, speech_token_len)
        # add noise for robustness
        noise_scale = 0.1
        emo_noise = torch.randn_like(emotion_token) * noise_scale
        emotion_token = emotion_token + emo_noise
        # masking
        emotion_token = emotion_token.masked_fill(~emotion_mask.transpose(1, 2), 0)

        # a.emotion loss
        # query
        emotion_query = torch.tensor(range(0, 9)).to(device)
        emotion_query = self.emotion_query_embedding(emotion_query).unsqueeze(0).repeat(speech_token.size(0), 1, 1) # [B, 9, 512]
        # cross attention
        emotion_key_padding_mask = make_pad_mask(speech_token_len, emotion_token.size(1)) # [B, L]
        emotion_feature, _ = self.emotion_score_encoder(query=emotion_query, key=emotion_token, value=emotion_token, key_padding_mask=emotion_key_padding_mask) # [B, L, 512] -> [B, 9, 512]
        # pooling

        emotion_score = torch.mean(self.global_emotion_head(emotion_feature), dim=-1) # [B, 9, 512] -> [B, 9, 128] -> [B, 9]
        # compute loss
        emotion_loss = self.global_emo_criterion(emotion_score, emotion)
        emo_sim = F.cosine_similarity(F.softmax(emotion_score.detach(), dim=1), emotion, dim=1).mean()

        # b.vad loss
        frame_features = self.word_pre_pool_net(emotion_token) # [B, L, D] -> [B, L, D]
        # iterate over the batch
        total_vad_loss = 0.
        valid_sample_count = 0
        # vad_sim = 0.
        max_len = emotion_token.size(1)
        for b in range(emotion_token.size(0)):
            # word alignment info of the current utterance [N_words, 5]
            words_info = align_words[b]
            word_features_list = []
            word_targets_list = []
            # iterate over each word
            for w_idx in range(len(words_info)):
                start_t, end_t, v, a, d = words_info[w_idx]
                # convert timestamps to frame indices
                start_i = int(start_t * self.token_frame_rate)
                end_i = int(end_t * self.token_frame_rate)
                # clamp boundaries
                start_i = max(0, start_i)
                end_i = min(max_len, end_i)
                # ensure a non-empty frame span (avoid index collision for very short words)
                if end_i > start_i:
                    # slice and pool
                    roi_frames = frame_features[b, start_i:end_i, :] # [T_segment, D]
                    word_vec = torch.mean(roi_frames, dim=0) # [D]
                    word_features_list.append(word_vec)
                    word_targets_list.append(torch.tensor([v, a, d], dtype=torch.float32).to(device))
            # batch computation (Linear -> VAD)
            num_words = len(word_features_list)
            if num_words > 0:
                # stack into a batch: [Sentence_Words, D]
                sentence_word_features = torch.stack(word_features_list)
                sentence_word_targets = torch.stack(word_targets_list)
                pred_word_vad = self.word_post_pool_net(sentence_word_features) # [Total_Words, 3]
                # compute loss
                sample_loss = 0.0
                # use CCC when there are >= 2 words
                if num_words >= 2:
                    # compute VAD loss
                    loss_v = CCCLoss(pred_word_vad[:, 0], sentence_word_targets[:, 0])
                    loss_a = CCCLoss(pred_word_vad[:, 1], sentence_word_targets[:, 1])
                    loss_d = CCCLoss(pred_word_vad[:, 2], sentence_word_targets[:, 2])
                    # CCCLoss may return None when variance is 0 (even with num_words >= 2, identical values give zero variance)
                    if loss_v is None or loss_a is None or loss_d is None:
                         sample_loss = self.mse_loss(pred_word_vad, sentence_word_targets)
                    else:
                         sample_loss = loss_v + loss_a + loss_d
                else:
                    # too few words to compute correlation, fall back to MSE
                    sample_loss = self.mse_loss(pred_word_vad, sentence_word_targets)
                total_vad_loss += sample_loss
                # vad_sim += F.cosine_similarity(pred_word_vad.detach(), sentence_word_targets, dim=1).mean()
                valid_sample_count += 1
            # accumulate loss
            if valid_sample_count > 0:
                vad_loss = total_vad_loss / valid_sample_count
                # vad_sim = vad_sim / valid_sample_count
            else:
                vad_loss = torch.tensor(0.0).to(device)
                # vad_sim = 0


        # 4. content token: ASR supervision
        # alignment
        content_token, content_mask = self.content_adapter(content_token, speech_token_len)
        content_token = content_token.masked_fill(~content_mask.transpose(1, 2), 0)
        # content token
        content_token = self.whisper_affine_layer(content_token)
        # asr loss
        asr_loss, word_er, _, _ = self.content_token_encoder(content, content_token, speech_token_len)

        # 5. get spk embedding
        if self.spk is False:
            embedding = torch.zeros(speech_token.size(0), 1, self.hidden_size).to(device)
        else:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(1)
        
        # 6. combine speech_token
        # a. add noise
        # weak noise
        emotion_low_noise = torch.randn_like(emotion_token_gn) / (8 * (self.emotion_fsq_scale - 1))
        content_low_noise = torch.randn_like(content_token_gn) / (8 * (self.content_fsq_scale - 1))
        # strong noise
        emotion_high_noise = torch.randn_like(emotion_token_gn) / (6 * (self.emotion_fsq_scale - 1))
        content_high_noise = torch.randn_like(content_token_gn) / (6 * (self.content_fsq_scale - 1))
        # mask
        emotion_mask = torch.rand_like(emotion_low_noise[:, :, :1], device=device) < random.uniform(0, 0.4) # [B, L, 1]
        content_mask = torch.rand_like(content_low_noise[:, :, :1], device=device) < random.uniform(0, 0.6) # [B, L, 1]
        emotion_low_noise = emotion_low_noise * ~emotion_mask
        emotion_high_noise = emotion_high_noise * emotion_mask
        content_low_noise = content_low_noise * ~content_mask
        content_high_noise = content_high_noise * content_mask
        # quantized results without noise
        _, pt_ori = self.FSQ(emotion_token_gn, 'emotion')
        _, ct_ori = self.FSQ(content_token_gn, 'content')
        # quantized results with noise
        emotion_token_gn = emotion_token_gn + emotion_low_noise + emotion_high_noise
        content_token_gn = content_token_gn + content_low_noise + content_high_noise
        emotion_token_gn, pt_tar = self.FSQ(emotion_token_gn, 'emotion')
        content_token_gn, ct_tar = self.FSQ(content_token_gn, 'content')
        # codebook consistency before vs. after the noise
        pt_simr = (pt_ori == pt_tar).float().mean()
        ct_simr = (ct_ori == ct_tar).float().mean()

        # b. get speech token
        emotion_token_gn = self.emotion_token_affine_linear(emotion_token_gn)
        content_token_gn = self.content_token_affine_linear(content_token_gn)
        # direct summation
        # content_token_gn, _ = self.modulation_adapter(content_token_gn, speech_token_len)
        # speech_token = (emotion_token_gn + content_token_gn) / 2
        # feature modulation
        style_params = self.modulation_layer(emotion_token_gn)
        gamma, beta = style_params.chunk(2, dim=-1) # split the last dim into two halves
        speech_token = content_token_gn * (1 + gamma) + beta
        # combiner
        speech_token = torch.cat((embedding, speech_token), dim=1)
        speech_token, _ = self.combiner(speech_token, speech_token_len + 1)
        speech_token = self.output_affine_linear(speech_token[:, 1:, :]) # [B, L, 6561]
        # compute loss
        re_loss = self.criterion(speech_token.view(-1, self.speech_token_size), target_token.view(-1))

        # 7. weighted sum of losses
        re_loss = re_loss*4
        # loss = re_loss + asr_loss
        emotion_loss = emotion_loss*0.2
        loss = re_loss + emotion_loss + vad_loss

        re_token = torch.argmax(speech_token, dim=2) # [B, L]
        re_token[token_mask] = IGNORE_ID

        acc = self.accelerate(re_token.clone(), target_token.clone())

        # return {'loss': loss, 're_loss': re_loss, 'asr_loss': asr_loss,
        #         'acc': acc, 'err': word_er, 'pt_sr': pt_simr, 'ct_sr': ct_simr}

        return {'loss': loss, 're_loss': re_loss, 'emotion_loss': emotion_loss, 'vad_loss': vad_loss,
                'acc': acc, 'emo_sim': emo_sim, 'err': word_er, 'pt_sr': pt_simr, 'ct_sr': ct_simr}

    def generate_token(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        

        # 1. prepare target: target_token for accuracy comparison,
        # true_feature for the loss
        target_token = speech_token.clone()
        token_mask = torch.arange(target_token.size(1)).to(device) >= speech_token_len.unsqueeze(1)
        target_token = target_token.masked_fill(token_mask, IGNORE_ID)

        # 2. decouple speech_token
        speech_token = self.speech_token_embedding(speech_token)
        emotion_token, emotion_token_mask = self.emotion_decoupler(speech_token, speech_token_len) # [B, L, 512]
        content_token, content_token_mask = self.content_decoupler(speech_token, speech_token_len) # [B, L, 512]

        emotion_token = emotion_token.masked_fill(~emotion_token_mask.transpose(1, 2), 0)
        content_token = content_token.masked_fill(~content_token_mask.transpose(1, 2), 0)

        # 3. get preference token
        emotion_token_gn = torch.sigmoid(self.emotion_FSQ_affine_linear(emotion_token))
        content_token_gn, _ = self.modulation_adapter(content_token.detach(), speech_token_len) # refine the acoustic skeleton
        content_token_gn = torch.sigmoid(self.content_FSQ_affine_linear(content_token_gn))

        emotion_token_gn, emotion_indexs = self.FSQ(emotion_token_gn, 'emotion')
        content_token_gn, content_indexs = self.FSQ(content_token_gn, 'content')

        return emotion_indexs, content_indexs

    def generate_reward(
            self,
            speech_token_emb: torch.Tensor,
            truth_emotion_token: torch.Tensor,
            truth_content_token: torch.Tensor,
            batch: dict,
            device: torch.device,
    ):
        """
        Args:
            speech_token_emb: (B, L, D)
            speech_token_len: (B)
            emotion: (B, 9)
        Outs:
            prompt_token: (L)
            content_token: (L)
        """
        speech_token_len = batch['speech_token_len'].to(device)
        emotion = batch['emotion'].to(device)
        content = batch['text']
        align_words = batch['align_words']

        speech_token_emb = F.linear(speech_token_emb, self.speech_token_embedding.weight.t())

        # 1. decouple speech_token
        emotion_token, emotion_token_mask = self.emotion_decoupler(speech_token_emb, speech_token_len) # [B, L, 512]
        content_token, content_token_mask = self.content_decoupler(speech_token_emb, speech_token_len) # [B, L, 512]

        emotion_token = emotion_token.masked_fill(~emotion_token_mask.transpose(1, 2), 0)
        content_token = content_token.masked_fill(~content_token_mask.transpose(1, 2), 0)

        # project down for reconstruction
        emotion_token_gn = torch.sigmoid(self.emotion_FSQ_affine_linear(emotion_token))
        content_token_gn, _ = self.modulation_adapter(content_token.detach(), speech_token_len) # refine the acoustic skeleton
        content_token_gn = torch.sigmoid(self.content_FSQ_affine_linear(content_token_gn))

        # 2. emotion token: hierarchical surpvise
        # alignment
        emotion_token, emotion_mask = self.emotion_adapter(emotion_token, speech_token_len)
        # masking
        emotion_token = emotion_token.masked_fill(~emotion_mask.transpose(1, 2), 0)

        # a.emotion loss
        # query
        emotion_query = torch.tensor(range(0, 9)).to(device)
        emotion_query = self.emotion_query_embedding(emotion_query).unsqueeze(0).repeat(speech_token_emb.size(0), 1, 1) # [B, 9, 512]
        # cross attention
        emotion_key_padding_mask = make_pad_mask(speech_token_len, emotion_token.size(1)) # [B, L]
        emotion_feature, _ = self.emotion_score_encoder(query=emotion_query, key=emotion_token, value=emotion_token, key_padding_mask=emotion_key_padding_mask) # [B, L, 512] -> [B, 9, 512]
        
        # pooling
        emotion_score = torch.mean(self.global_emotion_head(emotion_feature), dim=-1) # [B, 9, 512] -> [B, 9, 128] -> [B, 9]

        emotion_loss = self.global_emo_criterion(emotion_score, emotion)
        

        emotion_score = F.softmax(emotion_score.detach(), dim=1)
        emo_sim = F.cosine_similarity(emotion_score, emotion, dim=1).mean()

        # b.vad loss
        frame_features = self.word_pre_pool_net(emotion_token) # [B, L, D] -> [B, L, D]
        # iterate over the batch
        total_vad_loss = 0.
        valid_sample_count = 0
        # vad_sim = 0.
        max_len = emotion_token.size(1)
        for b in range(emotion_token.size(0)):
            # word alignment info of the current utterance [N_words, 5]
            words_info = align_words[b]
            word_features_list = []
            word_targets_list = []
            # iterate over each word
            for w_idx in range(len(words_info)):
                start_t, end_t, v, a, d = words_info[w_idx]
                # convert timestamps to frame indices
                start_i = int(start_t * self.token_frame_rate)
                end_i = int(end_t * self.token_frame_rate)
                # clamp boundaries
                start_i = max(0, start_i)
                end_i = min(max_len, end_i)
                # ensure a non-empty frame span (avoid index collision for very short words)
                if end_i > start_i:
                    # slice and pool
                    roi_frames = frame_features[b, start_i:end_i, :] # [T_segment, D]
                    word_vec = torch.mean(roi_frames, dim=0) # [D]
                    word_features_list.append(word_vec)
                    word_targets_list.append(torch.tensor([v, a, d], dtype=torch.float32).to(device))
            # batch computation (Linear -> VAD)
            num_words = len(word_features_list)
            if num_words > 0:
                # stack into a batch: [Sentence_Words, D]
                sentence_word_features = torch.stack(word_features_list)
                sentence_word_targets = torch.stack(word_targets_list)
                pred_word_vad = self.word_post_pool_net(sentence_word_features) # [Total_Words, 3]
                # compute loss
                sample_loss = 0.0
                # use CCC when there are >= 2 words
                if num_words >= 2:
                    # compute VAD loss
                    loss_v = CCCLoss(pred_word_vad[:, 0], sentence_word_targets[:, 0])
                    loss_a = CCCLoss(pred_word_vad[:, 1], sentence_word_targets[:, 1])
                    loss_d = CCCLoss(pred_word_vad[:, 2], sentence_word_targets[:, 2])
                    # CCCLoss may return None when variance is 0 (even with num_words >= 2, identical values give zero variance)
                    if loss_v is None or loss_a is None or loss_d is None:
                         sample_loss = self.mse_loss(pred_word_vad, sentence_word_targets)
                    else:
                         sample_loss = loss_v + loss_a + loss_d
                else:
                    # too few words to compute correlation, fall back to MSE
                    sample_loss = self.mse_loss(pred_word_vad, sentence_word_targets)
                total_vad_loss += sample_loss
                # vad_sim += F.cosine_similarity(pred_word_vad.detach(), sentence_word_targets, dim=1).mean()
                valid_sample_count += 1
            # accumulate loss
            if valid_sample_count > 0:
                vad_loss = total_vad_loss / valid_sample_count
                # vad_sim = vad_sim / valid_sample_count
            else:
                vad_loss = torch.tensor(0.0).to(device)
                # vad_sim = 0


        # 3. content token: ASR surpvise
        # alignment
        content_token, content_mask = self.content_adapter(content_token, speech_token_len)
        content_token = content_token.masked_fill(~content_mask.transpose(1, 2), 0)
        # content token
        content_token = self.whisper_affine_layer(content_token)
        # asr loss
        asr_loss, word_er, _, _ = self.content_token_encoder(content, content_token, speech_token_len)

        # 4. frame token
        max_len = emotion_token_gn.size(1)
        mask = torch.arange(max_len).to(device).unsqueeze(0) < speech_token_len.unsqueeze(1)
        emotion_token_loss = F.mse_loss(emotion_token_gn[mask], truth_emotion_token[mask])
        _, pred_emotion_idx = self.FSQ(emotion_token_gn, 'emotion')
        _, truth_emotion_idx = self.FSQ(truth_emotion_token, 'emotion')

        content_token_loss = F.mse_loss(content_token_gn[mask], truth_content_token[mask])
        _, pred_content_idx = self.FSQ(content_token_gn, 'content')
        _, truth_content_idx = self.FSQ(truth_content_token, 'content')

        # compute FSQ similarity
        truth_emotion_idx = truth_emotion_idx.masked_fill(~mask, IGNORE_ID)
        truth_content_idx = truth_content_idx.masked_fill(~mask, IGNORE_ID)
        
        emotion_token_sim = self.accelerate(pred_emotion_idx, truth_emotion_idx)
        content_token_sim = self.accelerate(pred_content_idx, truth_content_idx)
    
        return emotion_loss, vad_loss, asr_loss, emotion_token_loss, content_token_loss, emo_sim, word_er, emotion_token_sim, content_token_sim

    @torch.inference_mode()
    def inference(self, batch, device='cuda', language='zh', max_new_tokens=256, align_words=None):
        """
        Args:
            align_words: word-level alignment info of the current sample
                (inference batch size is 1). Accepts a list of dicts
                {word, start, end} or the training format
                [start_t, end_t, v, a, d].
        """
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        max_len = speech_token.size(1)
        valid_mask = torch.arange(max_len, device=device).unsqueeze(0) < speech_token_len.unsqueeze(1)

        token_emb = self.speech_token_embedding(speech_token)
        emotion_feature, emotion_mask = self.emotion_decoupler(token_emb, speech_token_len)
        content_feature, content_mask = self.content_decoupler(token_emb, speech_token_len)
        emotion_feature = emotion_feature.masked_fill(~emotion_mask.transpose(1, 2), 0)
        content_feature = content_feature.masked_fill(~content_mask.transpose(1, 2), 0)

        emotion_code = torch.sigmoid(self.emotion_FSQ_affine_linear(emotion_feature))
        content_code, _ = self.modulation_adapter(content_feature, speech_token_len)
        content_code = torch.sigmoid(self.content_FSQ_affine_linear(content_code))
        emotion_code, emotion_index = self.FSQ(emotion_code, 'emotion')
        content_code, content_index = self.FSQ(content_code, 'content')

        emotion_hidden, emotion_hidden_mask = self.emotion_adapter(emotion_feature, speech_token_len)
        emotion_hidden = emotion_hidden.masked_fill(~emotion_hidden_mask.transpose(1, 2), 0)
        emotion_query = torch.arange(9, device=device)
        emotion_query = self.emotion_query_embedding(emotion_query).unsqueeze(0).expand(speech_token.size(0), -1, -1)
        emotion_key_padding_mask = ~valid_mask
        emotion_query_feature, _ = self.emotion_score_encoder(
            query=emotion_query,
            key=emotion_hidden,
            value=emotion_hidden,
            key_padding_mask=emotion_key_padding_mask
        )
        emotion_logits = self.global_emotion_head(emotion_query_feature).mean(dim=-1)
        emotion_prob = F.softmax(emotion_logits, dim=-1)
        emotion_class = emotion_prob.argmax(dim=-1)

        frame_features = self.word_pre_pool_net(emotion_hidden)

        # frame-level VAD
        frame_vad = self.word_post_pool_net(frame_features)
        frame_vad = frame_vad.masked_fill(~valid_mask.unsqueeze(-1), 0)

        # sentence-level VAD: average-pool valid frames first,
        # then apply the VAD head
        sentence_feature = (frame_features * valid_mask.unsqueeze(-1)).sum(dim=1)
        sentence_feature = sentence_feature / valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
        sentence_vad = self.word_post_pool_net(sentence_feature)

        # word-level VAD: same as the vad branch in training forward,
        # mean-pool frame features over each aligned word's timestamp
        # span, then apply the VAD head
        word_vad = None
        word_vad_index = None
        if align_words is not None:
            max_feat_len = frame_features.size(1)
            word_features_list = []
            word_index_list = []
            for w_idx in range(len(align_words)):
                word_info = align_words[w_idx]
                # support both dict {word, start, end} and the training format [start_t, end_t, v, a, d]
                if isinstance(word_info, dict):
                    start_t, end_t = word_info['start'], word_info['end']
                else:
                    start_t, end_t = word_info[0], word_info[1]
                # convert timestamps to frame indices
                start_i = int(start_t * self.token_frame_rate)
                end_i = int(end_t * self.token_frame_rate)
                # clamp boundaries
                start_i = max(0, start_i)
                end_i = min(max_feat_len, end_i)
                # ensure a non-empty frame span (avoid index collision for very short words)
                if end_i > start_i:
                    # slice and pool
                    roi_frames = frame_features[0, start_i:end_i, :] # [T_segment, D]
                    word_vec = torch.mean(roi_frames, dim=0) # [D]
                    word_features_list.append(word_vec)
                    word_index_list.append(w_idx)
            # batch computation (Linear -> VAD)
            if len(word_features_list) > 0:
                sentence_word_features = torch.stack(word_features_list) # [N_words, D]
                word_vad = self.word_post_pool_net(sentence_word_features) # [N_words, 3]
                word_vad_index = torch.tensor(word_index_list, dtype=torch.long)

        content_hidden, content_hidden_mask = self.content_adapter(content_feature, speech_token_len)
        content_hidden = content_hidden.masked_fill(~content_hidden_mask.transpose(1, 2), 0)
        content_hidden = self.whisper_affine_layer(content_hidden)
        asr_text = self.content_token_encoder.transcribe(
            content_hidden,
            encoder_attention_mask=valid_mask.long(),
            language=language,
            max_new_tokens=max_new_tokens
        )

        return {
            'emotion_token': [emotion_index[i, :speech_token_len[i]].cpu() for i in range(speech_token.size(0))],
            'content_token': [content_index[i, :speech_token_len[i]].cpu() for i in range(speech_token.size(0))],
            'emotion_code': [emotion_code[i, :speech_token_len[i]].cpu() for i in range(speech_token.size(0))],
            'content_code': [content_code[i, :speech_token_len[i]].cpu() for i in range(speech_token.size(0))],
            'frame_vad': [frame_vad[i, :speech_token_len[i]].cpu() for i in range(speech_token.size(0))],
            'sentence_vad': sentence_vad.cpu(),
            'word_vad': word_vad.cpu() if word_vad is not None else None,
            'word_vad_index': word_vad_index.cpu() if word_vad_index is not None else None,
            'emotion_logits': emotion_logits.cpu(),
            'emotion_probability': emotion_prob.cpu(),
            'emotion_class': emotion_class.cpu(),
            'asr_text': asr_text
        }