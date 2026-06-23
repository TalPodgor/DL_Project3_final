"""Self-contained copy of the CUT / pix2pixHD generator used by the final model.

The final model `chess_v5_bright_silABC` builds its generator with the cloned
CUT framework's `networks.define_G(g_in, g_out, ngf=64, netG='resnet_9blocks',
normG='instance', use_dropout=False, no_antialias=False, no_antialias_up=False)`
(see `v5_work/v5_cluster_src/paired_geom_hd_model.py` and
`v5_work/final_config/bright_silABC_train_opt.txt`).

This module reproduces that exact architecture — the antialiased ResNet
generator (Zhang 2019 blur down/up-sampling) with `InstanceNorm2d(affine=False)`
— so the trained `latest_net_G.pth` `state_dict` loads here without needing the
full framework on the path. The `nn.Sequential` ordering is identical to the
upstream `ResnetGenerator`, so the parameter/buffer keys (`model.1.weight`,
`model.7.filt`, `model.12.conv_block.1.weight`, ...) match the checkpoint.

NOTE: the load is verified by `build_generator(...).load_state_dict(sd,
strict=True)` at runtime; an architecture mismatch raises loudly rather than
silently mis-loading. If you already have the cloned CUT repo on PYTHONPATH you
may instead use its `networks.define_G` — both yield the same keys.
"""
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_filter(filt_size=3):
    if filt_size == 1:
        a = np.array([1., ])
    elif filt_size == 2:
        a = np.array([1., 1.])
    elif filt_size == 3:
        a = np.array([1., 2., 1.])
    elif filt_size == 4:
        a = np.array([1., 3., 3., 1.])
    elif filt_size == 5:
        a = np.array([1., 4., 6., 4., 1.])
    elif filt_size == 6:
        a = np.array([1., 5., 10., 10., 5., 1.])
    else:
        a = np.array([1., 6., 15., 20., 15., 6., 1.])
    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)
    return filt


def get_pad_layer(pad_type):
    if pad_type in ["refl", "reflect"]:
        return nn.ReflectionPad2d
    if pad_type in ["repl", "replicate"]:
        return nn.ReplicationPad2d
    if pad_type == "zero":
        return nn.ZeroPad2d
    raise ValueError(f"Pad type [{pad_type}] not recognized")


class Downsample(nn.Module):
    def __init__(self, channels, pad_type="reflect", filt_size=3, stride=2, pad_off=0):
        super().__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))] * 2
        self.pad_sizes = [p + pad_off for p in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer("filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample(nn.Module):
    def __init__(self, channels, pad_type="repl", filt_size=4, stride=2):
        super().__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=filt_size) * (stride ** 2)
        self.register_buffer("filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(
            self.pad(inp), self.filt, stride=self.stride,
            padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if self.filt_odd:
            return ret_val
        return ret_val[:, :, :-1, :-1]


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """Antialiased ResNet generator, identical layout to CUT's `ResnetGenerator`."""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d,
                 use_dropout=False, n_blocks=9, padding_type="reflect",
                 no_antialias=False, no_antialias_up=False):
        assert n_blocks >= 0
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            if no_antialias:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def build_generator(in_ch=21, out_ch=3, ngf=64, n_blocks=9):
    """Construct the final generator (instance norm, antialiased, 9 ResNet blocks)."""
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    return ResnetGenerator(in_ch, out_ch, ngf=ngf, norm_layer=norm_layer,
                           use_dropout=False, n_blocks=n_blocks,
                           no_antialias=False, no_antialias_up=False)
