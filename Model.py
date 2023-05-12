# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 17:16:09 2022

@author: AmirPouya Hemmasian (ahemmasi@andrew.cmu.edu)
"""
import torch
from torch import nn
import numpy as np

class EncoderBlock(nn.Module):

    def __init__(self, c_in, c_out, mode='avg', activation=nn.LeakyReLU,
                 padding_mode='circular'):

        super().__init__()
        if mode == 'stride':
            layers = [nn.Conv2d(c_in, c_out, kernel_size=4, stride=2,
                                padding=1, padding_mode=padding_mode)
                      ]
        elif mode == 'avg':
            layers = [nn.Conv2d(c_in, c_out, kernel_size=3, stride=1,
                                padding=1, padding_mode=padding_mode),
                      nn.AvgPool2d(kernel_size=2)
                      ]
        elif mode == 'max':
            layers = [nn.Conv2d(c_in, c_out, kernel_size=3, stride=1,
                                padding=1, padding_mode=padding_mode),
                      nn.MaxPool2d(kernel_size=2)
                      ]
        layers += [activation()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):

        return self.net(x)


class Encoder(nn.Module):

    def __init__(self, channels=[1, 16, 32, 64, 128], pool_mode='avg',
                 activation=nn.LeakyReLU, padding_mode='circular'):

        super().__init__()
        cs = channels
        layers = []
        for i in range(len(cs)-1):
            layers += [EncoderBlock(cs[i], cs[i+1], pool_mode, activation,
                                    padding_mode)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):

        return self.net(x)


class DecoderBlock(nn.Module):

    def __init__(self, c_in, c_out, mode='bilinear', activation=nn.LeakyReLU,
                 padding_mode='circular'):

        super().__init__()
        if mode == 'stride':
            layers = [nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2,
                                         padding=1, padding_mode=padding_mode)
                      ]
        else:
            layers = [
                nn.Upsample(scale_factor=2, mode=mode, align_corners=False),
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1,
                          padding_mode=padding_mode)
                ]
        layers += [activation()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):

        return self.net(x)


class Decoder(nn.Module):

    def __init__(self, channels=[128, 64, 32, 16, 1], up_mode='bilinear',
                 activation=nn.LeakyReLU, padding_mode='circular'):

        super().__init__()
        cs = channels
        layers = []
        for i in range(len(cs)-1):
            act = nn.Identity if i == len(cs)-2 else activation
            layers += [DecoderBlock(cs[i], cs[i+1], up_mode, act,
                                    padding_mode)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):

        return self.net(x)


class Transformer2D_Layer(nn.Module):

    def __init__(self, embed_dim=128, num_heads=8,
                 kdim=None, vdim=None, hidden_dim=256):

        super().__init__()
        self.MHA = nn.MultiheadAttention(embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         kdim=kdim, vdim=vdim,
                                         batch_first=True)
        self.FF = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, 1),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dim, embed_dim, 1)
        )

    def forward(self, x):

        #  x has shape (B, C, N) where N=HW
        x = x.permute(0, 2, 1)
        # now, x has shape (B, N, C) where C=embed_dim
        x = x + self.MHA(x, x, x, need_weights=False)[0]
        x = x.permute(0, 2, 1)
        x = x + self.FF(x)
        return x


class Transformer2D(nn.Module):

    def __init__(self, shape=(4, 4), n_layers=6,
                 MHA_kwargs=dict(embed_dim=128, num_heads=8,
                                 kdim=None, vdim=None, hidden_dim=256),
                 periodic=True):

        super().__init__()
        self.spatial_shape = shape  # (nx, ny)
        nx, ny = shape

        if periodic:
            x, y = torch.meshgrid(torch.arange(nx), torch.arange(ny))
            x_freq = torch.fft.rfftfreq(nx)[1:, None, None]
            y_freq = torch.fft.rfftfreq(ny)[1:, None, None]
            x_sin = torch.sin(2*np.pi*x_freq*x)
            x_cos = torch.cos(2*np.pi*x_freq*x)
            y_sin = torch.sin(2*np.pi*y_freq*y)
            y_cos = torch.cos(2*np.pi*y_freq*y)
            pos_info = torch.cat([x_sin, x_cos, y_sin, y_cos])
        else:
            x, y = torch.meshgrid(torch.arange(1, nx+1)/nx,
                                  torch.arange(1, ny+1)/ny)
            pos_info = torch.stack([x, y])

        dim_pos = pos_info.shape[0]
        self.pos_info = pos_info.unsqueeze(0) # for the batch dimension

        self.pos_embedder = nn.Sequential(
            nn.Conv2d(dim_pos, dim_pos*4, 1), nn.LeakyReLU(),
            nn.Conv2d(dim_pos*4, MHA_kwargs['embed_dim'], 1)
            )

        layers = [Transformer2D_Layer(**MHA_kwargs) for i in range(n_layers)]
        self.transformer = nn.Sequential(*layers)

    def forward(self, x):

        x += self.pos_embedder(self.pos_info.to(x.device))
        x = x.flatten(-2)
        x = self.transformer(x)
        x = x.reshape(*x.shape[:-1], *self.spatial_shape)
        return x
