
import os
from os.path import join as ospj
import json
import glob
from shutil import copyfile

from tqdm import tqdm
import ffmpeg

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils

def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


@torch.no_grad()
def translate_reconstruction(nets, model, args, img_src, lat, lab_trg, filename, device):
    N = img_src.size(0)
    if args.latent_type == "wp":
        num_layers = 18 if args.gan_base == "ffhq" else 14
    else:
        num_layers = 1

    out_concat = [img_src]
    for i in range(3):
        if args.use_vae:
            rndz = torch.randn(img_src.size(0), num_layers, args.latent_dim).to(device).detach()
        else:
            rndz = torch.randn(img_src.size(0), args.rnd_dim).to(device).detach()
        lat_out = nets.mapnet(lat, rndz, lab_trg)
        img_trg = model(lat_out)
        out_concat.append(img_trg)
    out_concat = torch.cat(out_concat, dim=2)
    save_image(out_concat, N, filename)
    del out_concat        


@torch.no_grad()
def debug_image(nets, model, args, inputs, step, device):
    img_src, lat, lab_trg = inputs.img_src, inputs.lat, inputs.lab_trg
    filename = ospj(args.sample_dir, '%06d_rec.jpg' % (step))
    translate_reconstruction(nets, model, args, img_src, lat, lab_trg, filename, device)


# ======================= #
# Video-related functions #
# ======================= #


def tensor2ndarray255(images):
    images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    return images.cpu().numpy().transpose(0, 2, 3, 1) * 255