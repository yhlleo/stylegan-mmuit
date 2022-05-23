import math
import copy
import numpy as np
from munch import Munch

import torch
from torch import nn
import torch.nn.functional as F

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False, upsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.upsample   = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, num_features, norm_type='adaln'):
        super().__init__()
        if norm_type == "adaln":
            self.norm = nn.LayerNorm(num_features)
        else:
            self.norm = nn.InstanceNorm1d(num_features)
        self.fc = nn.Linear(num_features, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        gamma, beta = torch.chunk(h, chunks=2, dim=len(h.size())-1)
        return (1 + gamma) * self.norm(x) + beta


class ZNetwork(nn.Module):
    def __init__(
        self, 
        rnd_dim=64, 
        num_domains=4, 
        latent_dim=512, 
        latent_type='wp', 
        pre_norm=False
    ):
        super(ZNetwork, self).__init__()
        self.latent_type = latent_type

        layers = []
        layers += [
            nn.Linear(rnd_dim+num_domains, 512),
            nn.ReLU()
        ]

        for _ in range(3):
            if pre_norm:
                layers += [
                    nn.LayerNorm(512),
                    nn.Linear(512, 512),
                    nn.ReLU(inplace=True)
                ]
            else:
                layers += [
                    nn.Linear(512, 512),
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(512)
                ]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        num_layers = 18 if self.latent_type == "wp" else 1
        for _ in range(num_layers):
            self.unshared += [nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512)
            )]

    def forward(self, z, y):
        h = self.shared(torch.cat([z, y],dim=1))
        out = []
        for block in self.unshared:
            out += [block(h)]
        out = torch.stack(out, dim=1)  # (batch, num_layers, style_dim)
        if self.latent_type != "wp":
            out = out.squeeze(1)
        return out

class WNetwork(nn.Module):
    def __init__(
        self, 
        rnd_dim=64, 
        num_domains=4, 
        latent_dim=512, 
        latent_type='wp', 
        norm_type="adaln", 
        pre_norm=False,
    ):
        super(WNetwork, self).__init__()
        self.latent_type = latent_type

        self.z_net = ZNetwork(
            rnd_dim=rnd_dim,
            num_domains=num_domains,
            latent_dim=latent_dim,
            latent_type=latent_type,
            pre_norm=pre_norm
        )

        self.adain = None
        if norm_type in ['adain', 'adaln']:
            self.adain = AdaIN(num_features=latent_dim, norm_type=norm_type)

        if pre_norm:
            self.fc = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim)
            )

    def forward(self, x, z, y):
        h = self.z_net(z, y)
        if self.adain is not None:
            h = self.adain(x, h)
        else:
            h = h + x
        h = self.fc(h)
        return h

class MappingNetwork(nn.Module):
    def __init__(self,
        rnd_dim=64, 
        num_domains=4,
        latent_dim=512, 
        latent_type='wp',
        use_residual=False,
        norm_type='adaln',
        pre_norm=False
    ):   
        super(MappingNetwork, self).__init__()
        self.use_residual = use_residual
        self.w_net = WNetwork(
            rnd_dim, 
            num_domains, 
            latent_dim, 
            latent_type, 
            norm_type, 
            pre_norm
        )

    def forward(self, w_org, z, y):
        w_out = self.w_net(w_org, z, y)
        if self.use_residual:
            w_out += w_org
        return w_out

class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=4, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out


        self.main = nn.Sequential(*blocks)
        self.out_cls = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_out, dim_out, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_out, num_domains, 1, 1, 0))

        self.out_src = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_out, dim_out, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_out, 1, 1, 1, 0))

    def forward(self, x, y=None):
        out = self.main(x)
        out_cls = self.out_cls(out).view(x.size(0), -1)  # (batch, num_domains)
        out_src = self.out_src(out).view(x.size(0))
        return out_cls, out_src

def build_model(args, is_distributed=True, device=None):
    mapping_net = MappingNetwork(
        rnd_dim=args.rnd_dim,
        num_domains=args.num_domains,
        latent_dim=args.latent_dim,
        latent_type=args.latent_type,
        use_residual=args.use_residual,
        norm_type=args.norm_type,
        pre_norm=args.pre_norm
    ).to(device)
    discriminator = Discriminator(
        img_size=args.img_size, 
        num_domains=args.num_domains
    ).to(device)

    mapping_net_ema = copy.deepcopy(mapping_net)
    discriminator_ema = copy.deepcopy(discriminator)

    if is_distributed:
        mapping_net = nn.parallel.DistributedDataParallel(
            mapping_net,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False
        )
        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
    nets = Munch(
        mapnet=mapping_net,
        discriminator=discriminator)
    nets_ema = Munch(
        mapnet=mapping_net_ema,
        discriminator=discriminator)

    return nets, nets_ema

def build_model_single(args):
    mapping_net = MappingNetwork(
        rnd_dim=args.rnd_dim,
        num_domains=args.num_domains,
        latent_dim=args.latent_dim,
        latent_type=args.latent_type,
        use_residual=args.use_residual,
        norm_type=args.norm_type,
        pre_norm=args.pre_norm
    )
    discriminator = Discriminator(
        img_size=args.img_size, 
        num_domains=args.num_domains
    )
    nets = Munch(
        mapnet=mapping_net,
        discriminator=discriminator)
    return nets
