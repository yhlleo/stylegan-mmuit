import os.path
import argparse
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torchvision.utils as vutils

from models import StyleGAN2Generator

boundary_dict = {
  0: "stylegan2_ffhq_gende.npy",
  1: "stylegan2_ffhq_eyeglasses.npy",
  2: "stylegan2_ffhq_age.npy",
  3: "stylegan2_ffhq_smile.npy"
}

def main():
  """Main function."""
  mode = "test"
  boundary_path = "/apdcephfs/share_916081/amosyhliu/stylegan-mmuit/interfacegan-boundaries"
  if mode == "val":
    input_latent_codes_path = "../datasets/stylegan-mmuit2/val/styleganv1-ffhq/styleflow-styleganv1-latents-w.npy"
  elif mode == "test":
    input_latent_codes_path = "/apdcephfs/share_916081/yliu/datasets/styleflow/test/wp.npy"
  else:
    input_latent_codes_path = "../datasets/celebahq-test/styleganv1-celebahq-latents-w.npy"
  
  save_dir = "/apdcephfs/share_916081/amosyhliu/stylegan-mmuit/interfacegan-stylegan2-interp"
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  model = StyleGAN2Generator()
  model._load_pretrain("/apdcephfs/share_916081/amosyhliu/pretrained_models/stylegan2-ffhq-config-f.pt")
  model.eval().cuda()

  latent_codes = np.load(input_latent_codes_path)
  latent_codes = torch.from_numpy(latent_codes).float().cuda()
  total_num = latent_codes.shape[0]

  for idx in [0]:
    boundary = np.load(os.path.join(boundary_path, boundary_dict[idx]))
    boundary = torch.from_numpy(boundary).float().unsqueeze(1).repeat(1, 18, 1).cuda()
    print("The loaded boundary shape:", boundary.size())
    for i in tqdm(range(total_num)):
      latent_codes_pos = latent_codes[i,:] + 3.0 * boundary 
      latent_codes_neg = latent_codes[i,:] - 3.0 * boundary

      for idy, alpha in enumerate(np.arange(0., 1., 0.05)):
        s = torch.lerp(latent_codes_pos, latent_codes_neg, alpha)
        fake = model.forward_test(s)

        fake = torch.clamp(fake * 0.5 + 0.5, 0, 1)
        vutils.save_image(fake.data, os.path.join(save_dir, '{:04d}-{}.jpg'.format(i, idy)), padding=0)

if __name__ == '__main__':
  main()
