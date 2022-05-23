# python3.7
"""Trains semantic boundary from latent space.

Basically, this file takes a collection of `latent code - attribute score`
pairs, and find the separation boundary by treating it as a bi-classification
problem and training a linear SVM classifier. The well-trained decision boundary
of the SVM classifier will be saved as the boundary corresponding to a
particular semantic from the latent space. The normal direction of the boundary
can be used to manipulate the correpsonding attribute of the synthesis.
"""

import os.path
import argparse
import numpy as np

from utils.logger import setup_logger
from utils.manipulator import train_boundary

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Train semantic boundary with given latent codes and '
                  'attribute scores.')
  parser.add_argument('-o', '--output_dir', type=str, required=True,
                      help='Directory to save the output results. (required)')
  parser.add_argument('-c', '--latent_codes_path', type=str, required=True,
                      help='Path to the input latent codes. (required)')
  parser.add_argument('-s', '--scores_path', type=str, required=True,
                      help='Path to the input attribute scores. (required)')
  parser.add_argument('-n', '--chosen_num_or_ratio', type=float, default=0.02,
                      help='How many samples to choose for training. '
                           '(default: 0.2)')
  parser.add_argument('-r', '--split_ratio', type=float, default=0.7,
                      help='Ratio with which to split training and validation '
                           'sets. (default: 0.7)')
  parser.add_argument('-V', '--invalid_value', type=float, default=None,
                      help='Sample whose attribute score is equal to this '
                           'field will be ignored. (default: None)')
  parser.add_argument('--index', type=int, default=0)

  return parser.parse_args()

boundary_dict = {
  0: "stylegan2_ffhq_gende",
  1: "stylegan2_ffhq_eyeglasses",
  2: "stylegan2_ffhq_age",
  3: "stylegan2_ffhq_smile"
}

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

def main():
  """Main function."""
  args = parse_args()
  logger = setup_logger(args.output_dir, logger_name='generate_data', logfile_name='{}.txt'.format(boundary_dict[args.index]))

  logger.info('Loading latent codes.')
  if not os.path.isfile(args.latent_codes_path):
    raise ValueError(f'Latent codes `{args.latent_codes_path}` does not exist!')
  #latent_codes = np.load(args.latent_codes_path)
  latent_codes = np.load(args.latent_codes_path)[:,0,0] # [N, 512]

  logger.info('Loading attribute scores.')
  selected_attrs = ["Gender", "Glasses", "Age", "Expression"]
  if not os.path.isfile(args.scores_path):
    raise ValueError(f'Attribute scores `{args.scores_path}` does not exist!')
  all_attributes = preprocess(args.scores_path, selected_attrs)
  scores = np.array(all_attributes)[:, args.index:args.index+1]
  
  boundary = train_boundary(latent_codes=latent_codes,
                            scores=scores,
                            chosen_num_or_ratio=args.chosen_num_or_ratio,
                            split_ratio=args.split_ratio,
                            invalid_value=args.invalid_value,
                            logger=logger)
  np.save(os.path.join(args.output_dir, '{}.npy'.format(boundary_dict[args.index])), boundary)


if __name__ == '__main__':
  main()
