import math

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


UPSAMPLE_STRIDE2KERNEL_DICT = {(1, 2): (1, 5), (1, 4): (1, 7), (2, 1): (5, 1), (2, 2): (3, 3)}
UPSAMPLE_STRIDE2PAD_DICT = {(1, 2): (2, 2, 0, 0), (1, 4): (3, 3, 0, 0), (2, 1): (0, 0, 2, 2), (2, 2): (1, 1, 1, 1)}


class CircularConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        if 'padding' in kwargs:
            self.is_pad = True
            if isinstance(kwargs['padding'], int):
                h1 = h2 = v1 = v2 = kwargs['padding']
            elif isinstance(kwargs['padding'], tuple):
                h1, h2, v1, v2 = kwargs['padding']
            else:
                raise NotImplementedError
            self.h_pad, self.v_pad = (h1, h2, 0, 0), (0, 0, v1, v2)
            del kwargs['padding']
        else:
            self.is_pad = False

        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.is_pad:
            if sum(self.h_pad) > 0:
                x = nn.functional.pad(x, self.h_pad, mode="circular")  # horizontal pad
            if sum(self.v_pad) > 0:
                x = nn.functional.pad(x, self.v_pad, mode="constant")  # vertical pad
        x = self._conv_forward(x, self.weight, self.bias)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, stride):
        super().__init__()
        self.with_conv = with_conv
        self.stride = stride
        if self.with_conv:
            k, p = UPSAMPLE_STRIDE2KERNEL_DICT[stride], UPSAMPLE_STRIDE2PAD_DICT[stride]
            self.conv = CircularConv2d(in_channels, in_channels, kernel_size=k, padding=p)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.stride, mode='bilinear', align_corners=True)
        if self.with_conv:
            x = self.conv(x)
        return x


DOWNSAMPLE_STRIDE2KERNEL_DICT = {(1, 2): (3, 3), (1, 4): (3, 5), (2, 1): (3, 3), (2, 2): (3, 3)}
DOWNSAMPLE_STRIDE2PAD_DICT = {(1, 2): (0, 1, 1, 1), (1, 4): (1, 1, 1, 1), (2, 1): (1, 1, 1, 1), (2, 2): (0, 1, 0, 1)}


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, stride):
        super().__init__()
        self.with_conv = with_conv
        self.stride = stride
        if self.with_conv:
            k, p = DOWNSAMPLE_STRIDE2KERNEL_DICT[stride], DOWNSAMPLE_STRIDE2PAD_DICT[stride]
            self.conv = CircularConv2d(in_channels, in_channels, kernel_size=k, stride=stride, padding=p)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)  # modified for lidar
        return x


UNIFORM_KERNEL2PAD_DICT = {(3, 3): (1, 1, 1, 1), (1, 4): (1, 2, 0, 0)}


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, kernel_size=(3, 3), conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        pad = UNIFORM_KERNEL2PAD_DICT[kernel_size]

        self.norm1 = Normalize(in_channels)
        self.conv1 = CircularConv2d(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=pad)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CircularConv2d(out_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=pad)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CircularConv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=1,
                                                    padding=pad)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    # print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class Encoder(nn.Module):
    def __init__(self, *, ch=128, out_ch=3, ch_mult=(1, 1, 2, 2, 4), strides=[[1,2],[1,2],[1,2],[1,2]], num_res_blocks=2,
                 attn_levels=[4], dropout=0.0, resamp_with_conv=True, in_channels=3, z_channels=16,
                 double_z=True, use_linear_attn=False, attn_type="vanilla", use_mask=False,
                 **ignore_kwargs):
        super().__init__()
        if use_mask:
            assert out_ch == in_channels + 1, 'Set "out_ch = out_ch + 1" for mask prediction.'
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # downsampling
        self.conv_in = CircularConv2d(in_channels,
                                      self.ch,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if i_level in attn_levels:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                stride = tuple(strides[i_level])
                down.downsample = Downsample(block_in, resamp_with_conv, stride)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = CircularConv2d(block_in,
                                       2 * z_channels if double_z else z_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch=128, out_ch=3, ch_mult=(1, 1, 2, 2, 4), strides=[[1,2],[1,2],[1,2],[1,2]], num_res_blocks=2, attn_levels=[4],
                 dropout=0.0, resamp_with_conv=True, in_channels=3, z_channels=16, give_pre_end=False,
                 tanh_out=False, use_linear_attn=False, attn_type="vanilla", use_mask=False,
                 **ignorekwargs):
        super().__init__()
        stride2kernel = {(2, 2): (3, 3), (1, 2): (1, 4)}
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = CircularConv2d(z_channels,
                                      block_in,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            stride = tuple(strides[i_level - 1]) if i_level > 0 else None
            kernel = stride2kernel[stride] if stride is not None else (1, 4)
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         kernel_size=kernel,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if i_level in attn_levels:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if stride is not None:
                up.upsample = Upsample(block_in, resamp_with_conv, stride)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = CircularConv2d(block_in,
                                       out_ch,
                                       kernel_size=(1, 4),
                                       stride=1,
                                       padding=(1, 2, 0, 0))

    def forward(self, z):
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
