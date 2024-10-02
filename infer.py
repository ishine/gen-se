import fire
import torch
import torchaudio
import yaml

from models.gense import N2S, S2S

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_firstchannel_read(path, target_sr=16000):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav[0].unsqueeze(0)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
    return wav.unsqueeze(0)


def run(noisy_path, out_path, config_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = AttrDict(config)

    noisy_wav = get_firstchannel_read(noisy_path).to(device)

    n2s_model = N2S(config)
    n2s_model.load_state_dict(torch.load(config.path['n2s_ckpt_path'])["state_dict"])
    n2s_model = n2s_model.eval()
    n2s_model = n2s_model.to(device)

    s2s_model = S2S(config)
    s2s_model.load_state_dict(torch.load(config.path['s2s_ckpt_path'])["state_dict"])
    s2s_model = s2s_model.eval()
    s2s_model = s2s_model.to(device)

    noisy_s, clean_s = n2s_model.generate(noisy_wav)
    enhanced_wav = s2s_model.generate(noisy_wav, noisy_s, clean_s)
    torchaudio.save(out_path, enhanced_wav, sample_rate=16000)
    

if __name__ == "__main__":
    fire.Fire(
        {
            "run": run,
        }
    )