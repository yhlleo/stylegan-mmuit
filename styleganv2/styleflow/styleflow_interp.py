
import os
import sys
import torch
import argparse
import pickle

import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils

sys.path.append(".")
from module.flow import cnf
from models.stylegan2.model import Generator

from styleflow.basic import load_data, subset_selection

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='./styleflow-results-interp')
parser.add_argument('--checkpoint_path', type=str, default='./pretrained_models/stylegan2-ffhq-config-f.pt')
parser.add_argument('--use_selection', type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda')
ckpt = torch.load(args.checkpoint_path, map_location='cpu')['g_ema']
gan_model = Generator(1024, 512, 8).to(device).eval()
gan_model.load_state_dict(ckpt, strict=True)

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

#attr_order = ['Gender', 'Glasses', 'Yaw', 'Pitch', 'Baldness', 'Beard', 'Age', 'Expression']
prior = cnf(512, '512-512-512-512-512', 17, 1)
prior.load_state_dict(torch.load('flow_weight/modellarge10k.pt'))
prior.eval()

all_attr, all_light, all_ws = load_data()
attr_edit_dict = {
    0: [0.0, 1.0],
    1: [0.0, 1.0],
    6: [20.0, 65.0],
    7: [0.0, 1.0]
}

for i in tqdm(range(1000)):
    attr  = torch.from_numpy(all_attr[i]).unsqueeze(0).unsqueeze(-1).float().to(device) 
    light = torch.from_numpy(all_light[i]).float().to(device)
    w_src = torch.from_numpy(all_ws[i]).float().to(device)

    final_array_target = torch.cat([light, attr], dim=1)
    zero_padding = torch.zeros(1, 18, 1).to(device)
    for idx in [0]:
        xmin, xmax = attr_edit_dict[idx]
        rev0 = subset_selection(
            w=w_src, 
            arr=final_array_target, 
            index=idx, 
            value=xmin, 
            model=prior, 
            zero_padding=zero_padding,
            use_selection=args.use_selection)
        rev1 = subset_selection(
            w=w_src, 
            arr=final_array_target, 
            index=idx, 
            value=xmax, 
            model=prior, 
            zero_padding=zero_padding,
            use_selection=args.use_selection)

        w1 = rev0[0]
        w2 = rev1[0]

        for idy, alphq in enumerate(np.arange(0., 1., 0.05)):
            s = torch.lerp(w1, w2, alphq)
            GAN_image, _ = gan_model([s], input_is_latent=True, randomize_noise=True)
            GAN_image = torch.clamp(GAN_image * 0.5 + 0.5, 0, 1)
            vutils.save_image(GAN_image.data, os.path.join(args.save_dir, '{:04d}-{}.jpg'.format(i, idy)), padding=0)


