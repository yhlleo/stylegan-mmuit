import os.path
import argparse
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torchvision.utils as vutils

from models.model_settings import MODEL_POOL
from models.pggan_generator2 import PGGANGenerator
from models.stylegan_generator2 import StyleGANGenerator
from utils.logger import setup_logger

boundary_dict = {
  0: "stylegan_ffhq_gender_w_boundary.npy",
  1: "stylegan_ffhq_eyeglasses_w_boundary.npy",
  2: "stylegan_ffhq_age_w_boundary.npy",
  3: "stylegan_ffhq_smile_w_boundary.npy"
}

def main():
  """Main function."""
  logger = setup_logger('./', logger_name='eval_stylegan')
  boundary_path = "./boundaries"
  input_latent_codes_path = "/path/to/datasets/stylegan-ffhq/test/wp.npy"
  
  save_dir = "interfacegan-results"
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  model_name = "stylegan_ffhq"
  gan_type = MODEL_POOL[model_name]['gan_type']
  if gan_type == 'pggan':
    model = PGGANGenerator(model_name, logger)
  elif gan_type == 'stylegan':
    model = StyleGANGenerator(model_name, logger)
  else:
    raise NotImplementedError(f'Not implemented GAN type `{gan_type}`!')

  latent_codes = np.load(input_latent_codes_path)
  latent_codes = torch.from_numpy(latent_codes).float().cuda()
  latent_codes = model.preprocess(latent_codes, latent_space_type='w')
  print("The loaded latent codes shape:", latent_codes.size())
  total_num = min(1000, latent_codes.shape[0])

  model.model.eval().cuda()

  for idx in [0,1,2,3]:
    boundary = np.load(os.path.join(boundary_path, boundary_dict[idx]))
    boundary = torch.from_numpy(boundary).float().cuda()
    print("The loaded boundary shape:", boundary.size())
    for i in tqdm(range(total_num)):
      latent_codes_pos = latent_codes[i:i+1,:] + 3.0 * boundary 
      latent_codes_neg = latent_codes[i:i+1,:] - 3.0 * boundary
      fake_pos = model.synthesize(latent_codes_pos, latent_space_type='w')['image']
      fake_neg = model.synthesize(latent_codes_neg, latent_space_type='w')['image']

      fake_pos = torch.clamp(fake_pos * 0.5 + 0.5, 0, 1)
      fake_neg = torch.clamp(fake_neg * 0.5 + 0.5, 0, 1)

      vutils.save_image(fake_pos.data, os.path.join(save_dir, '{:04d}-{}-1.jpg'.format(i, idx)), padding=0)
      vutils.save_image(fake_neg.data, os.path.join(save_dir, '{:04d}-{}-0.jpg'.format(i, idx)), padding=0)

if __name__ == '__main__':
  main()
