import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.nn import Conv1d, ConvTranspose1d

LRELU_SLOPE = 0.1
alpha = 1.0

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)
        self.num_layers = len(self.convs1) + len(self.convs2) # total number of conv layers
        self.activations = nn.ModuleList([nn.LeakyReLU(LRELU_SLOPE) for _ in range(self.num_layers)])


    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2,a1,a2 in zip(self.convs1, self.convs2,acts1,acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class Encoder(torch.nn.Module):
    def __init__(self, h):
        super(Encoder, self).__init__()
        self.n_filters = h.en_filters
        self.vq_dim = h.vq_dim
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.upsample_initial_channel = self.n_filters * ( 2**self.num_upsamples )
        self.conv_pre = weight_norm(Conv1d(h.channel, self.n_filters, 7, 1, padding=3))
        self.normalize = nn.ModuleList()
        resblock = ResBlock1 

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(list(reversed(list(zip(h.upsample_rates, h.upsample_kernel_sizes))))):
            self.ups.append(weight_norm(
                Conv1d(self.n_filters*(2**i), self.n_filters*(2**(i+1)),
                       k, u,
                       padding=((k-u)//2)
                )))
        self.resblocks = nn.ModuleList()
        ch = 1
        for i in range(len(self.ups)):
            ch = self.n_filters*(2**(i+1))
            for j, (k, d) in enumerate(
                    zip(
                        list(reversed(h.resblock_kernel_sizes)),
                        list(reversed(h.resblock_dilation_sizes))
                    )
            ):
                self.resblocks.append(resblock(h, ch, k, d))
                self.normalize.append(torch.nn.LayerNorm([ch],eps=1e-6,elementwise_affine=True))
        
        self.activation_post = nn.LeakyReLU(LRELU_SLOPE)
        self.conv_post = Conv1d(ch, self.vq_dim, 3, 1, padding=1)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                    xs = self.normalize[i*self.num_kernels+j](xs.transpose(1,2)).transpose(1,2)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
                    xs = self.normalize[i*self.num_kernels+j](xs.transpose(1,2)).transpose(1,2)
            x = xs / self.num_kernels
        x = self.activation_post(x)
        x = self.conv_post(x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)

class Quantizer_module(torch.nn.Module):
    def __init__(self, n_e, e_dim):
        super(Quantizer_module, self).__init__()
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)
        self.target = torch.arange(0,n_e)

    def forward(self, x, idx=0):
        loss=torch.Tensor([0.0])
        d = torch.sum(x ** 2, 1, keepdim=True) + torch.sum(self.embedding.weight ** 2, 1) \
            - 2 * torch.matmul(x, self.embedding.weight.T)
        min_indicies = torch.argmin(d, 1)
        z_q = self.embedding(min_indicies)
        embed_vec = self.embedding.weight
        embed_dis = torch.mm(embed_vec , embed_vec.T)*3
        self.target = torch.arange(0,embed_vec.shape[0]).to(x.device)
        loss = F.cross_entropy(embed_dis,self.target)*(idx==0)
        return z_q, min_indicies,loss

class Quantizer(torch.nn.Module):
    def __init__(self, h):
        super(Quantizer, self).__init__()
        assert h.vq_dim % h.n_code_groups == 0
        self.lm_offset = 0
        self.lm_states = None
        self.vq_dim = h.vq_dim
        self.residul_layer = h.n_q
        self.n_code_groups = h.n_code_groups
        self.quantizer_modules = nn.ModuleList()
        for i in range(self.residul_layer):
            self.quantizer_modules.append(nn.ModuleList([
                Quantizer_module(h.n_codes, self.vq_dim // h.n_code_groups) for _ in range(h.n_code_groups)
            ]))
        self.h = h
        self.codebook_loss_lambda = self.h.codebook_loss_lambda     # e.g., 1
        self.commitment_loss_lambda = self.h.commitment_loss_lambda # e.g., 0.25
        

    def for_one_step(self, xin, idx):
        xin = xin.transpose(1, 2)
        x = xin.reshape(-1, self.vq_dim)
        x = torch.split(x, self.vq_dim // self.h.n_code_groups, dim=-1)
        min_indicies = []
        z_q = []
        all_losses = []
        for _x, m in zip(x, self.quantizer_modules[idx]):
            _z_q, _min_indicies,_loss = m(_x,idx)
            all_losses.append(_loss)
            z_q.append(_z_q)
            min_indicies.append(_min_indicies) 
        z_q = torch.cat(z_q, -1).reshape(xin.shape)
        z_q = z_q.transpose(1, 2)
        all_losses = torch.stack(all_losses)
        loss = torch.mean(all_losses)
        return z_q, min_indicies, loss
        
    
    def forward(self, xin,bw=-1,mask_id=None):
        quantized_out = 0.0
        residual = xin
        all_losses = []
        all_indices = []
        if bw<=0:
            bw = self.residul_layer
        for i in range(bw):
            quantized,  indices, e_loss = self.for_one_step(residual, i) # 
            if mask_id is not None:
                mask = (
                    torch.full([xin.shape[0],xin.shape[2],1], fill_value=i, device=xin.device) < mask_id.unsqueeze(2) + 1
                )
                mask = mask.repeat(1,1,xin.shape[1]).transpose(1,2)
            if mask_id is not None:
                loss = 0.1 * e_loss + self.codebook_loss_lambda * torch.mean((quantized - residual.detach()) ** 2 * mask) \
                    + self.commitment_loss_lambda * torch.mean((quantized.detach() - residual) ** 2 * mask ) 
            else:
                loss = 0.1 * e_loss \
                    +  self.codebook_loss_lambda * torch.mean((quantized - residual.detach()) ** 2 ) \
                    + self.commitment_loss_lambda * torch.mean((quantized.detach() - residual) ** 2  ) 
                
            quantized = residual + (quantized - residual).detach()
            residual = residual - quantized
            if mask_id is not None:
                quantized_out = quantized_out + quantized * mask
            else:
                quantized_out = quantized_out + quantized
            all_indices.extend(indices) # 
            all_losses.append(loss)
        all_losses = torch.stack(all_losses)
        loss = torch.mean(all_losses)
        return quantized_out, loss, all_indices
    
    def embed(self, x , bw=-1):
        quantized_out = torch.tensor(0.0, device=x.device)
        x = torch.split(x, 1, 2) 
        if bw <= 0 or bw > self.residul_layer:
            bw = self.residul_layer
        for i in range(bw):
            ret = []
            for j in range(self.n_code_groups):
                q = x[j+self.n_code_groups*i]
                embed = self.quantizer_modules[i][j]
                q = embed.embedding(q.squeeze(-1))
                ret.append(q)
            ret = torch.cat(ret, -1)
            quantized_out = quantized_out + ret
        return quantized_out.transpose(1, 2)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.n_filters = h.de_filters
        self.vq_dim = h.vq_dim
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.upsample_initial_channel = self.n_filters * ( 2**self.num_upsamples )
        self.conv_pre = weight_norm(Conv1d(self.vq_dim, self.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1
        

        self.norm = nn.Identity()

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(
                    self.upsample_initial_channel//(2**i), self.upsample_initial_channel//(2**(i+1)),
                                k, u,
                                padding=(k - u )//2,
                )
            ))
        ch = 1
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = self.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))
        
        
        self.activation_post = nn.LeakyReLU(LRELU_SLOPE)
        self.conv_post = weight_norm(Conv1d(ch, h.channel, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv_pre(x)
        
        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)