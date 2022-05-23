
import os
import sys
import glob
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.autograd import Variable

sys.path.append(".")
sys.path.append("..")
from metrics.id_loss import IDLoss

parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=str)
parser.add_argument('--trg_dir', type=str)
parser.add_argument('--index', type=int, default=0)
args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])])

source_list = glob.glob(os.path.join(args.src_dir, '*.jpg'))
source_list.sort()
target_list = glob.glob(os.path.join(args.trg_dir, '*-{}-*.jpg'.format(args.index)))
target_list.sort()
device = torch.device('cuda')
frs = IDLoss().to(device).eval()

frs_pos_scores = []
frs_neg_scores = []

all_images = [v.split('/')[-1].split('.')[0] for v in source_list]

for im in tqdm(range(len(all_images))):
    src_img = Variable(transform(Image.open(os.path.join(args.src_dir, "{}.jpg".format(all_images[im]))).convert("RGB")).unsqueeze(0).to(device))
    trg_pos = Variable(transform(Image.open(os.path.join(args.trg_dir, "{:04d}-{}-1.jpg".format(im, args.index))).convert("RGB")).unsqueeze(0).to(device))
    trg_neg = Variable(transform(Image.open(os.path.join(args.trg_dir, "{:04d}-{}-0.jpg".format(im, args.index))).convert("RGB")).unsqueeze(0).to(device))

    frs_pos_scores.append(frs.cal_identity_similarity(src_img, trg_pos))
    frs_neg_scores.append(frs.cal_identity_similarity(src_img, trg_neg))

frs_pos_scores = np.array(frs_pos_scores)
frs_neg_scores = np.array(frs_neg_scores)
print("Overal FRS:", (np.mean(frs_pos_scores) + np.mean(frs_neg_scores))/2.0)
print("Positive FRS:", np.mean(frs_pos_scores))
print("Negative FRS:", np.mean(frs_neg_scores))

