import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import timm
from timm.layers import trunc_normal_
from PIL import Image

from .blocks import Block
from .model_utils import adapt_input_conv, padding, unpadding, resize_pos_embed, init_weights
from .stems import PatchEmbedding, ConvStem
from .decoders import DecoderLinear, DecoderUpConv
from .rangevit_kpconv import RangeViT_KPConv, KPClassifier
from ..range_utils import pcd2range, range2xyz


def preprocess_range(range, **kwargs):
    depth_img, intensity = range
    xyz_img = range2xyz(depth_img, log_scale=False, **kwargs)
    depth_img = depth_img[None]
    intensity = intensity[None]
    img = np.vstack([depth_img, xyz_img, intensity])
    return img

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        n_cls,
        dropout=0.1,
        drop_path_rate=0.0,
        channels=3,
        ls_init_values=None,
        patch_stride=None,
        conv_stem='none',
        stem_base_channels=32,
        stem_hidden_dim=None,
    ):
        super().__init__()

        self.conv_stem = conv_stem

        if self.conv_stem == 'none':
            self.patch_embed = PatchEmbedding(
                image_size,
                patch_size,
                patch_stride,
                d_model,
                channels,)
        else:   # in this case self.conv_stem = 'ConvStem'
            assert patch_stride == patch_size # patch_size = patch_stride if a convolutional stem is used

            self.patch_embed = ConvStem(
                in_channels=channels,
                base_channels=stem_base_channels,
                img_size=image_size,
                patch_stride=patch_stride,
                embed_dim=d_model,
                flatten=True,
                hidden_dim=stem_hidden_dim)

        self.patch_size = patch_size
        self.PS_H, self.PS_W = patch_size
        self.patch_stride = patch_stride
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls
        self.image_size = image_size

        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.num_patches + 1, d_model))

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
                [Block(d_model, n_heads, d_ff, dropout, dpr[i], init_values=ls_init_values) for i in range(n_layers)]
            )

        self.norm = nn.LayerNorm(d_model)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_grid_size(self, H, W):
        return self.patch_embed.get_grid_size(H, W)

    def forward(self, im, return_features=False):
        B, _, H, W = im.shape
        x, skip = self.patch_embed(im) # x.shape = [16, 576, 384]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # x.shape = [16, 577, 384]

        pos_embed = self.pos_embed
        num_extra_tokens = 1

        if x.shape[1] != pos_embed.shape[1]:
            grid_H, grid_W = self.get_grid_size(H, W)
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (grid_H, grid_W),
                num_extra_tokens,
            )

        x = x + pos_embed
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x, skip  # x.shape = [16, 577, 384]


def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    model_cfg.pop('backbone')
    mlp_expansion_ratio = 4
    model_cfg['d_ff'] = mlp_expansion_ratio * model_cfg['d_model']

    new_patch_size = model_cfg.pop('new_patch_size')
    new_patch_stride = model_cfg.pop('new_patch_stride')

    if (new_patch_size is not None):
        if new_patch_stride is None:
            new_patch_stride = new_patch_size
        model_cfg['patch_size'] = new_patch_size
        model_cfg['patch_stride'] = new_patch_stride

    model = VisionTransformer(**model_cfg)

    return model


def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop('name')
    decoder_cfg['d_encoder'] = encoder.d_model
    decoder_cfg['patch_size'] = encoder.patch_size

    if name == 'linear':
        decoder_cfg['patch_stride'] = encoder.patch_stride
        decoder = DecoderLinear(**decoder_cfg)
    elif name == 'up_conv':
        decoder_cfg['patch_stride'] = encoder.patch_stride
        decoder = DecoderUpConv(**decoder_cfg)
    else:
        raise ValueError(f'Unknown decoder: {name}')
    return decoder

def create_rangevit(model_cfg, use_kpconv=False):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop('decoder')
    decoder_cfg['n_cls'] = model_cfg['n_cls']

    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)

    if use_kpconv:
        kpclassifier = KPClassifier(
            in_channels=decoder_cfg['d_decoder'] ,
            out_channels=decoder_cfg['d_decoder'],
            num_classes=model_cfg['n_cls'])
        model = RangeViT_KPConv(encoder, decoder, kpclassifier, n_cls=model_cfg['n_cls'])
    else:
        model = RangeViT_noKPConv(encoder, decoder, n_cls=model_cfg['n_cls'])

    return model

class RangeViT(nn.Module):
    def __init__(self, cfg):
        super(RangeViT, self).__init__()

        self.n_classes = cfg.n_classes
        self.in_channels = cfg.in_channels


        new_patch_size = cfg.patch_size 
        new_patch_stride = cfg.patch_stride
        decoder = cfg.decoder


        if cfg.vit_backbone == 'vit_small_patch16_384':
            n_heads = 6
            n_layers = 12
            patch_size = 16
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 384
        elif cfg.vit_backbone == 'vit_base_patch16_384':
            n_heads = 12
            n_layers = 12
            patch_size = 16
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 768
        elif cfg.vit_backbone == 'vit_large_patch16_384':
            n_heads = 16
            n_layers = 24
            patch_size = 16
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 1024
        else:
            raise NameError('Not known ViT backbone.')

        # Decoder config
        if decoder == 'linear':
            decoder_cfg = {'n_classes': self.n_classes, 'name': 'linear'}
        elif decoder == 'up_conv':
            decoder_cfg = {
                'n_cls': self.n_classes, 'name': 'up_conv',
                'd_decoder': 128, # hidden dim of the decoder
                'scale_factor': (2, 8), # scaling factor in the PixelShuffle layer
                'skip_filters': cfg.skip_filters,} # channel dim of the skip connection (between the convolutional stem and the up_conv decoder)

        # ViT encoder and stem config
        net_kwargs = {
            'backbone': cfg.vit_backbone,
            'd_model': d_model, # dim of features
            'decoder': decoder_cfg,
            'drop_path_rate': drop_path_rate,
            'dropout': dropout,
            'channels': cfg.in_channels, # nb of channels for the 3D point projections
            'image_size': cfg.image_size,
            'n_cls': self.n_classes,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'patch_size': patch_size, # old patch size for the ViT encoder
            'new_patch_size': new_patch_size, # new patch size for the ViT encoder
            'new_patch_stride': new_patch_stride, # new patch stride for the ViT encoder
            'conv_stem': cfg.conv_stem,
            'stem_base_channels': cfg.stem_base_channels,
            'stem_hidden_dim': cfg.D_h,
        }

        # Create RangeViT model
        self.rangevit = create_rangevit(net_kwargs, cfg.use_kpconv)

        self.pool = torch.nn.AvgPool2d(kernel_size=(8, 1024))

        image_mean = torch.tensor([11.274, 0.076, 0.246, -1.041, 0.215])[None,:,None,None] #KITTI360 dataset
        image_std = torch.tensor([9.816, 10.239, 7.616, 0.797, 0.181])[None,:,None,None]
        self.register_buffer('image_mean', image_mean) 
        self.register_buffer('image_std', image_std)

        self.proj_dict = {
            "fov": (cfg.sensor.fov_up, cfg.sensor.fov_down),
            "size": cfg.original_image_size,
            "depth_range": cfg.sensor.depth_range,
            "depth_scale": cfg.sensor.depth_scale,
        }


    def forward(self, x):
        x = [preprocess_range(pcd, **self.proj_dict) for pcd in x]
        x = torch.from_numpy(np.stack(x)).float().to(self.image_std.device)
        x = (x - self.image_mean) / self.image_std
        x = x[:,:self.in_channels]
        x = self.rangevit.forward_2d_features(x)
        x = self.pool(x)
        return x.flatten(1).cpu()