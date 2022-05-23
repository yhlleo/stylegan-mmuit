import os.path
import argparse
import cv2
import copy
import numpy as np
from tqdm import tqdm

import torch
import torchvision.utils as vutils

from models import StyleGAN2Generator
from core.model import MappingNetwork
from ours_edit import preprocess, post_process

parser = argparse.ArgumentParser()
# model arguments
parser.add_argument('--num_domains', type=int, default=4,
                    help='Number of domains')
parser.add_argument('--rnd_dim', type=int, default=64,
                    help='Latent vector dimension')
parser.add_argument('--latent_dim', type=int, default=512,
                    help='Latent code dimension of stylegan')
parser.add_argument('--latent_type', type=str, default='wp', help='[z, w, wp]')
parser.add_argument('--norm_type', type=str, default='adaln', help='adaln | adain | none')
parser.add_argument('--pre_norm', type=int, default=0)
parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
                    help='Directory for saving network checkpoints')
parser.add_argument('--resume_iter', type=int, default=50000,
                    help='Iterations to resume training/testing')
parser.add_argument('--use_post', type=int, default=0)
parser.add_argument('--group_index', type=int, default=-1)
parser.add_argument('--group_size', type=int, default=50)
parser.add_argument('--save_dir', type=str, default='./results-interp')
args = parser.parse_args()


attribute_dict = {
    0: "Gender",
    1: "Glasses",
    2: "Age",
    3: "Expression"
}

selected_attrs = ["Gender", "Glasses", "Age", "Expression"]

def main(args):
  """Main function."""
  attribute_path = "/path/to/datasets/styleflow/list_attr_ffhq-test.txt"
  input_latent_codes_path = "/path/to/datasets/styleflow/test/wp.npy"

  all_attributes = preprocess(attribute_path, selected_attrs)
  all_attributes = np.array(all_attributes)
  all_attributes = torch.from_numpy(all_attributes).unsqueeze(1).float().cuda() # [N, 1, D]

  model = StyleGAN2Generator()
  model._load_pretrain("/path/to/pretrained_models/stylegan2-ffhq-config-f.pt")
  model.eval().cuda()

  latent_codes = np.load(input_latent_codes_path)
  latent_codes = torch.from_numpy(latent_codes).float().cuda()
  total_num = latent_codes.shape[0]

  save_dir = args.save_dir #"ours-results-interp"
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  mapping_net = MappingNetwork(
    num_domains=args.num_domains,
    latent_dim=args.latent_dim,
    latent_type=args.latent_type,
    norm_type=args.norm_type,
    pre_norm=args.pre_norm
  )
  ckpt = torch.load(os.path.join(args.checkpoint_dir, "{:06d}_nets_ema.ckpt".format(args.resume_iter)), map_location='cpu')
  mapping_net.load_state_dict(ckpt['mapnet'])
  mapping_net.eval().cuda()

  for idx in [0]:
    if args.group_index > -1:
      start_idx = args.group_index*args.group_size
      end_idx = (args.group_index+1)*args.group_size
    else:
      start_idx = 0
      end_idx = 1000
    for i in tqdm(range(start_idx, end_idx)):
      lat, lab_src = latent_codes[i,:],  all_attributes[i]
      lab_trg_pos = copy.deepcopy(lab_src)
      lab_trg_neg = copy.deepcopy(lab_src)

      lab_trg_pos[:,idx] = 1.0
      lab_trg_neg[:,idx] = 0.0

      if args.use_vae:
        rndz = torch.randn(1, 18 if args.latent_type == "wp" else 1, args.latent_dim).cuda().detach()
      else:
        rndz = torch.randn(1, args.rnd_dim).cuda().detach()
      lat_pos = mapping_net(lat, rndz, lab_trg_pos)

      if args.use_vae:
        rndz = torch.randn(1, 18 if args.latent_type == "wp" else 1, args.latent_dim).cuda().detach()
      else:
        rndz = torch.randn(1, args.rnd_dim).cuda().detach()
      lat_neg = mapping_net(lat, rndz, lab_trg_neg)
      
      if args.use_post:
        lat_pos = post_process(lat_pos, lat, attribute_dict[idx])
        lat_neg = post_process(lat_neg, lat, attribute_dict[idx])
      
      for idy, alpha in enumerate(np.arange(0., 1., 0.05)):
        s = torch.lerp(lat_pos, lat_neg, alpha)
        fake = model.forward_test(s)

        fake = torch.clamp(fake * 0.5 + 0.5, 0, 1)
        vutils.save_image(fake.data, os.path.join(save_dir, '{:04d}-{}.jpg'.format(i, idy)), padding=0)

if __name__ == '__main__':
  main(args)
