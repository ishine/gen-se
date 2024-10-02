import torch
import torch.nn as nn
import joblib
from components.semantic_extractor.WavLM import WavLM, WavLMConfig

class ApplyKmeans(nn.Module):
    def __init__(self, km_path, device='cuda'):
        super(ApplyKmeans, self).__init__()
        print(f'Init k-means model from {km_path}')
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)
        self.C = torch.from_numpy(self.C_np).to(device)
        self.Cnorm = torch.from_numpy(self.Cnorm_np).to(device)
        self.emb = nn.Embedding(num_embeddings=300, embedding_dim=1024)
        self.emb.weight.data = self.C.transpose(0, 1)
        self.emb.weight.require_grad = False

    def forward(self, x, b, t):
        if not hasattr(self, 'C'):
            self.C = torch.from_numpy(self.C_np).to(x.device)
        if not hasattr(self, 'Cnorm'):
            self.Cnorm = torch.from_numpy(self.Cnorm_np).to(x.device)
        dist = x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
        tokens = dist.argmin(dim=-1).reshape(b, t)
        return tokens

def get_ssl_model(ckpt_path, km_path, device='cuda', type='xlsr'):
    if type == 'xlsr':
        print(f'Init xlsr model from {ckpt_path}')
        import fairseq
        import argparse
        task_arg = argparse.Namespace(task='audio_pretraining')
        task = fairseq.tasks.setup_task(task_arg)
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path], task=task)
        model = model[0]
        model.eval()
    elif type == 'wavlm':
        print(f'Init wavlm model from {ckpt_path}')
        cpt = torch.load(ckpt_path, map_location="cpu")
        cfg = WavLMConfig(cpt["cfg"])
        model = WavLM(cfg)
        model.load_state_dict(cpt["model"])
        model = model.eval()
        model = model.requires_grad_(False)
    else:
        raise NotImplementedError
    km_model = ApplyKmeans(km_path, device)
    return model, km_model

