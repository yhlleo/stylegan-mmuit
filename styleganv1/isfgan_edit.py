import os.path
import argparse
import cv2
import copy
import numpy as np
from tqdm import tqdm

import torch
import torchvision.utils as vutils

from models.model_settings import MODEL_POOL
from models.pggan_generator2 import PGGANGenerator
from models.stylegan_generator2 import StyleGANGenerator
from utils.logger import setup_logger

from core.model import MappingNetwork

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
parser.add_argument('--pre_norm', type=int, default=1)
parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
                    help='Directory for saving network checkpoints')
parser.add_argument('--resume_iter', type=int, default=50000,
                    help='Iterations to resume training/testing')
parser.add_argument('--gan_model_name', type=str, default='stylegan_ffhq', help="stylegan_ffhq | stylegan_celebahq")
parser.add_argument('--use_post', type=int, default=0)
parser.add_argument('--group_index', type=int, default=-1)
parser.add_argument('--group_size', type=int, default=50)
parser.add_argument('--save_dir', type=str, default='./results')
args = parser.parse_args()

def post_process(lat_bf, lat_af, attribute="Gender"):
    lat_new = copy.deepcopy(lat_bf)
    if attribute == "Gender":
        lat_new[:,2:4] = lat_af[:,2:4]
    elif attribute == "Age":
        lat_new[:,2:6] = lat_af[:,2:6]
    elif attribute == "Expression":
        lat_new[:,1:4] = lat_af[:,1:4]
    elif attribute == "Glasses":
        lat_new[:,1:4] = lat_af[:,1:4]
    else:
        pass
    return lat_new

attribute_dict = {
    0: "Gender",
    1: "Glasses",
    2: "Age",
    3: "Expression"
}

selected_attrs = ["Gender", "Glasses", "Age", "Expression"]

def preprocess(attr_path, selected_attrs):
  """Preprocess the CelebA attribute file."""
  image2attr = []
  attr2idx = {}
  lines = [line.rstrip() for line in open(attr_path, 'r')]
  all_attr_names = lines[1].split()
  for i, attr_name in enumerate(all_attr_names):
    attr2idx[attr_name] = i

  lines = lines[2:]
  for i, line in enumerate(lines):
    split = line.split()
    values = split[1:]

    label = []
    for attr_name in selected_attrs:
      idx = attr2idx[attr_name]
      label.append(int(values[idx] == '1'))

    image2attr.append(label)
  return image2attr


def main(args):
  """Main function."""
  attribute_path = "/path/to/datasets/stylegan-ffhq/list_attr_celeba-test.txt"
  input_latent_codes_path = "/path/to/datasets/stylegan-ffhq/test/wp.npy"


  all_attributes = preprocess(attribute_path, selected_attrs)
  all_attributes = np.array(all_attributes)
  all_attributes = torch.from_numpy(all_attributes).unsqueeze(1).float().cuda() # [N, 1, D]

  logger = setup_logger('./', logger_name='ours_eval_stylegan')
  model = StyleGANGenerator(args.gan_model_name, logger)
  model.model.eval().cuda()

  latent_codes = np.load(input_latent_codes_path)
  latent_codes = torch.from_numpy(latent_codes).float().cuda()
  latent_codes = model.preprocess(latent_codes, latent_space_type=args.latent_type)
  total_num = latent_codes.shape[0]

  save_dir = args.save_dir #"ours-results"
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  mapping_net = MappingNetwork(
    num_domains=args.num_domains,
    latent_dim=args.latent_dim,
    latent_type=args.latent_type,
    norm_type=args.norm_type,
    pre_norm=args.pre_norm,
    use_vae=args.use_vae
  )
  ckpt = torch.load(os.path.join(args.checkpoint_dir, "{:06d}_nets_ema.ckpt".format(args.resume_iter)), map_location='cpu')
  cur_state = mapping_net.state_dict()
  for key in cur_state:
    cur_state[key] = ckpt['mapnet']["module."+key]
  mapping_net.load_state_dict(cur_state)
  #mapping_net.load_state_dict(ckpt['mapnet'])
  mapping_net.eval().cuda()

  attributes = [0,1,2,3]

  for idx, idy in zip(attributes, [0,1,2,3]):
    if args.group_index > -1:
      start_idx = args.group_index*args.group_size
      end_idx = (args.group_index+1)*args.group_size
    else:
      start_idx = 0
      end_idx = 1000
    for i in tqdm(range(start_idx, end_idx)):
      lat, lab_src = latent_codes[i:i+1,:],  all_attributes[i]
      lab_trg_pos = copy.deepcopy(lab_src)
      lab_trg_neg = copy.deepcopy(lab_src)

      lab_trg_pos[:,idx] = 1.0
      lab_trg_neg[:,idx] = 0.0

      if args.use_vae:
        rndz = torch.randn(1, 18 if args.latent_type == "wp" else 1, args.latent_dim).cuda().detach()
      else:
        rndz = torch.randn(1, args.rnd_dim).cuda().detach()
      lat_pos = mapping_net(lat, rndz, lab_trg_pos)
      lat_pos = model.preprocess(lat_pos, latent_space_type=args.latent_type)
      if args.use_post:
        lat_pos = post_process(lat, lat_pos, attribute_dict[idx])
      fake_pos = model.synthesize(lat_pos, latent_space_type=args.latent_type)['image']

      if args.use_vae:
        rndz = torch.randn(1, 18 if args.latent_type == "wp" else 1, args.latent_dim).cuda().detach()
      else:
        rndz = torch.randn(1, args.rnd_dim).cuda().detach()
      lat_neg = mapping_net(lat, rndz, lab_trg_neg)
      lat_neg = model.preprocess(lat_neg, latent_space_type=args.latent_type)
      if args.use_post:
        lat_neg = post_process(lat, lat_neg, attribute_dict[idx])
      fake_neg = model.synthesize(lat_neg, latent_space_type=args.latent_type)['image']  

      fake_pos = torch.clamp(fake_pos * 0.5 + 0.5, 0, 1)
      fake_neg = torch.clamp(fake_neg * 0.5 + 0.5, 0, 1)

      vutils.save_image(fake_pos.data, os.path.join(save_dir, '{:04d}-{}-0.jpg'.format(i, idy)), padding=0)
      vutils.save_image(fake_neg.data, os.path.join(save_dir, '{:04d}-{}-1.jpg'.format(i, idy)), padding=0) 
if __name__ == '__main__':
  main(args)
