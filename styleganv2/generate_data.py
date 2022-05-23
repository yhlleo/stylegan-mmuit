import os
import copy
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torchvision.utils as vutils
from torch.nn import functional as F

from models import StyleGAN2Generator
import random

parser = argparse.ArgumentParser()
# model arguments
parser.add_argument('--save_dir', type=str)
parser.add_argument('--index', type=int, default=0)
args = parser.parse_args()

def save_image(img, save_dir, fname):
  vutils.save_image(img.data, os.path.join(save_dir, fname), padding=0)

def main(args):
  """Main function."""
  model = StyleGAN2Generator()
  model._load_pretrain("/path/to/stylegan2-ffhq-config-f.pt")
  model.eval().cuda()

  if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

  input_latent_codes_path = "/path/to/datasets/styleflow/train/wp.npy"
  latent_codes = np.load(input_latent_codes_path)
  latent_codes = torch.from_numpy(latent_codes).float().cuda()
  total_num = latent_codes.shape[0]

  latents = []

  img_id = 0
  with torch.no_grad():
    for i in tqdm(range(10000)):
      lat_src = latent_codes[i]
      img = model.forward_test(lat_src)
      img = torch.clamp(img * 0.5 + 0.5, 0.0, 1.0)
      latents.append(lat_src.cpu())

      save_image(img, args.save_dir, '{:06d}.png'.format(img_id))
      img_id += 1

      for _ in range(7):
        while True:
          rnd_id = random.randint(0, total_num-1)
          if rnd_id != i:
            break
        lat_ref = latent_codes[rnd_id]

        start_index = random.randint(0, 17)
        width = random.randint(2, 10)
        end_index = min(start_index+width, 18)
        
        lat_mix = copy.deepcopy(lat_src)
        lat_mix[:, start_index:end_index] = lat_ref[:, start_index:end_index]
        img = model.forward_test(lat_mix)
        img = torch.clamp(img * 0.5 + 0.5, 0.0, 1.0)
        latents.append(lat_mix.cpu())
        
        save_image(img, args.save_dir, '{:06d}.png'.format(img_id))
        img_id += 1
  latents = torch.stack(latents, dim=0).data.cpu().numpy()
  np.save(os.path.join(args.save_dir, "wp.npy"), latents)
 
if __name__ == '__main__':
  main(args)
