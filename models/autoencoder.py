"""
Ref:
    https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py
"""
from dictdot import dictdot
import torch
import numpy as np
import torch.nn as nn

from models import model_lidm, model_ldm


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape, device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        embed_dim,
        ch_mult,
        strides=None,
        use_variational=True,
        lib_name='ldm',
        out_ch=3,
        in_channels=3,
        attn_levels=[4],
        ):
        super().__init__()
        model_lib = eval(f'model_{lib_name}')
        self.encoder = model_lib.Encoder(ch_mult=ch_mult,
                                        z_channels=embed_dim,
                                        strides=strides,
                                        out_ch=out_ch,
                                        in_channels=in_channels,
                                        attn_levels=attn_levels)
        self.decoder = model_lib.Decoder(ch_mult=ch_mult,
                                        z_channels=embed_dim,
                                        strides=strides,
                                        out_ch=out_ch,
                                        in_channels=in_channels,
                                        attn_levels=attn_levels)
        self.use_variational = use_variational
        mult = 2 if self.use_variational else 1
        self.quant_conv = torch.nn.Conv2d(2 * embed_dim, mult * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, embed_dim, 1)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        if not self.use_variational:
            moments = torch.cat((moments, torch.ones_like(moments)), 1)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        # NOTE: We wrap the output in a dict to be consistent with the output
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dictdot(dict(sample=dec))

    def forward(self, x, return_recon=True):
        posterior = self.encode(x)
        z = posterior.sample()


        recon = None
        if return_recon:
            recon = self.decode(z).sample
        return posterior, z, recon


# Predefined VAE architectures
def VAE_F8D4(**kwargs):
    # [B, 4, 32, 32]
    return AutoencoderKL(embed_dim=4, ch_mult=[1, 2, 4, 4], use_variational=True, **kwargs)


def VAE_F16D32(**kwargs):
    # [B, 32, 16, 16] (used in VA-VAE and our model)
    return AutoencoderKL(embed_dim=32, ch_mult=[1, 1, 2, 2, 4], use_variational=True, **kwargs)

def VAE_H16D4(**kwargs):
    # [B, 4, 64, 64] (used in LIDM)
    return AutoencoderKL(embed_dim=4,
                        ch_mult=[1, 1, 2, 2, 4],
                        strides=[[1,2],[1,2],[1,2],[1,2]],
                        use_variational=True,
                        in_channels=1,
                        out_ch=1,
                        lib_name='lidm',
                        **kwargs)

def VAE_H16P2D16(**kwargs):
    # [B, 16, 32, 32] (used in LIDM)
    return AutoencoderKL(embed_dim=16,
                        ch_mult=[1, 1, 2, 2, 2, 4],
                        strides=[[1,2],[1,2],[1,2],[1,2],[2,2]],
                        use_variational=True,
                        in_channels=1,
                        out_ch=1,
                        lib_name='lidm',
                        **kwargs)

def VAE_H2P4D8(**kwargs):
    # [B, 8, 16, 128] (used in LIDM)
    return AutoencoderKL(embed_dim=8,
                        ch_mult=[1,2,2,4],
                        strides=[[1,2],[2,2],[2,2]],
                        use_variational=True,
                        in_channels=1,
                        out_ch=1,
                        lib_name='lidm',
                        attn_levels=[],
                        **kwargs)

def VAE_H1P4D4(**kwargs):
    # [B, 4, 16, 256] (used in LIDM)
    return AutoencoderKL(embed_dim=4,
                        ch_mult=[1,2,4],
                        strides=[[2,2],[2,2]],
                        use_variational=True,
                        in_channels=1,
                        out_ch=1,
                        lib_name='lidm',
                        attn_levels=[],
                        **kwargs)

vae_models = {
    "f8d4": VAE_F8D4, # [B, 4, 32, 32]
    "f16d32": VAE_F16D32, # [B, 32, 16, 16]
    "h16d4": VAE_H16D4, # [B, 4, 64, 64]
    "h16p2d16": VAE_H16P2D16, # [B, 16, 32, 32]
    "h2p4d8": VAE_H2P4D8, # [B, 8, 16, 128]
    "h1p4d4": VAE_H1P4D4, # [B, 4, 16, 256]
}