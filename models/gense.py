import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from components.semantic_extractor.ssl_model import get_ssl_model
from components.simcodec.model import SimCodec
from transformers import GPT2Config, GPT2LMHeadModel

class N2S(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        self.xlsr, self.km = get_ssl_model(**hps.ssl_model)
        self.bos = 1
        self.eos = 2
        self.pad = 0
        self.shift_num = 3

        self.lm_conf = GPT2Config(
            vocab_size=self.hps.model['n2s_vocab_size'],  
            n_embd=self.hps.model['hidden_size'],  
            n_layer=self.hps.model['num_hidden_layers'],  
            n_head=self.hps.model['num_attention_heads'],  
            activation_function='gelu_new',
            n_positions=2048,  
            n_ctx=2048, 
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-05,
            initializer_range=0.02,
            summary_type='mean',
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            bos_token_id=self.bos,
            eos_token_id=self.eos,
        )
        self.lm = GPT2LMHeadModel(self.lm_conf)
    
    def extract_semantic(self, wavs, num_frames):
        padding_size = (0, 100)
        wavs = F.pad(wavs, padding_size, "constant", 0)
        num_frames += 100
        features = self.xlsr.extract_features(wavs, padding_mask=None)
        layer_results = features['layer_results'][5]
        x, _, _ = layer_results
        features = x.transpose(0,1)
        b, t, d = features.shape
        tokens = self.km(features.reshape(-1, d), b=b, t=t)
        return tokens
    
    def inference(self, token_gen, pos_gen):
        predict_len = (token_gen.shape[1] - 1)
        truck_length = token_gen.shape[1]

        for j in tqdm(range(predict_len)):
            lm_outputs = self.lm(
                input_ids=token_gen,
                attention_mask=None,
                position_ids=pos_gen
            )
            logits = lm_outputs['logits']
            logits[:, :, 0:self.shift_num] = -1e5
            probs = logits[:, -1, :].softmax(dim=-1)

            dist = torch.distributions.categorical.Categorical(probs=probs)

            samples = dist.sample().unsqueeze(1).to(token_gen.device)
            token_gen = torch.cat([token_gen, samples], dim=1)
            pos_pad = torch.ones(pos_gen.shape[0]) * j
            pos_gen = torch.cat([pos_gen, pos_pad.unsqueeze(1).to(token_gen.device).long()], dim=1)
        
        return token_gen[:,truck_length:][0]


    def generate(self, mix):
        mix = mix.squeeze(1)
        num_frame = torch.LongTensor([mix.shape[1]]).to(mix.device)
        token_s = self.extract_semantic(mix, num_frames=num_frame)

        token_s += 3
        bos = torch.ones(token_s.shape[0],1).long().to(mix.device)
        token_gen = torch.cat([token_s, bos], dim=1)

        pos_gen_id = torch.from_numpy(np.asarray(list(range(token_s.shape[1] + 1)))).to(mix.device)
        pos_gen = []
        for i in range(token_s.shape[0]):
            pos_gen.append(pos_gen_id.unsqueeze(0))
        pos_gen = torch.cat(pos_gen, dim=0)

        clean_s = self.inference(token_gen, pos_gen) - self.shift_num
        token_s -= self.shift_num
        return token_s, clean_s


class S2S(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        self.codec_tokenizer = SimCodec(hps.path['codec_config_path'])
        self.xlsr, self.km = get_ssl_model(**hps.ssl_model)
        self.bos = 1
        self.eos = 2
        self.pad = 0
        self.shift_num = 3 + self.hps.model['semantic_num']
        self.lm_conf = GPT2Config(
            vocab_size=self.hps.model['s2s_vocab_size'],  
            n_embd=self.hps.model['hidden_size'],  
            n_layer=self.hps.model['num_hidden_layers'],  
            n_head=self.hps.model['num_attention_heads'],  
            activation_function='gelu_new',
            n_positions=4096,
            n_ctx=4096,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-05,
            initializer_range=0.02,
            summary_type='mean',
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            bos_token_id=self.bos,
            eos_token_id=self.eos,
        )
        self.lm = GPT2LMHeadModel(self.lm_conf)
    
    def inference(self, token_gen, pos_gen):
        predict_len = int((token_gen.shape[1] - 1) / 2)
        truck_length = token_gen.shape[1]
        for j in tqdm(range(predict_len)):
            lm_outputs = self.lm(
                input_ids=token_gen,
                attention_mask=None,
                position_ids=pos_gen
            )
            logits = lm_outputs['logits']
            logits[:, :, 0:self.shift_num] = -1e5
            probs = logits[:, -1, :].softmax(dim=-1)
            dist = torch.distributions.categorical.Categorical(probs=probs)
            samples = dist.sample().unsqueeze(1).to(token_gen.device)
            token_gen = torch.cat([token_gen, samples], dim=1)
            pos_pad = torch.ones(pos_gen.shape[0]) * (j + 1000)
            pos_gen = torch.cat([pos_gen, pos_pad.unsqueeze(1).to(token_gen.device).long()], dim=1)

        return token_gen[:,truck_length:][0]
    
    def generate(self, mix, mix_s, clean_s):
        mix_a = self.codec_tokenizer(mix).squeeze(-1)
        if len(clean_s.shape) == 1:
            clean_s = clean_s.unsqueeze(0)
        
        mix_s += 3
        clean_s += 3
        mix_a += self.shift_num

        bos = torch.ones(mix_s.shape[0],1).long().to(mix.device)
        token_gen = torch.cat([mix_s, clean_s, bos, mix_a], dim=1)

        pos_gen_id = torch.from_numpy(np.asarray(list(range(mix_s.shape[1] + clean_s.shape[1] + 1)) + list(range(mix_a.shape[1])))).to(mix.device)
        pos_gen = []
        for i in range(mix_s.shape[0]):
            pos_gen.append(pos_gen_id.unsqueeze(0))
        pos_gen = torch.cat(pos_gen, dim=0)

        pre_a = self.inference(token_gen, pos_gen) - self.shift_num
        gen_wav = self.codec_tokenizer.decode(pre_a.unsqueeze(0).unsqueeze(2)).squeeze(0).cpu()

        return gen_wav