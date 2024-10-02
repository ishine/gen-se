import json
import torch
import torch.nn as nn
from components.simcodec.modules import Encoder, Quantizer, Generator

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class SimCodec(nn.Module):
    def __init__(self, config_path):
        super(SimCodec, self).__init__()
        self.config_path = config_path
        with open(self.config_path) as f:
            data = f.read()
        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        self.encoder = Encoder(self.h)
        self.quantizer = Quantizer(self.h)
        self.generator = Generator(self.h)

    def forward(self, x):
        batch_size = x.size(0)
        if len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        c = self.encoder(x) 
        _, _, c = self.quantizer(c)
        c = [code.reshape(batch_size, -1) for code in c]
        return torch.stack(c, -1)

    def decode(self, x):
        return self.generator(self.quantizer.embed(x))