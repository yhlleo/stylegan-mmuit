
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

from metrics.lpips import LPIPS

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str)
parser.add_argument('--num_samples', type=int, default=10)
args = parser.parse_args()

device = torch.device('cuda')
lpips = LPIPS(device=device).eval().cuda()

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])])

all_imgs = glob.glob(os.path.join(args.img_dir, '*-*.jpg'))
all_imgs.sort()
all_names = []
for fpath in all_imgs:
    name = fpath.split('/')[-1].split('.')[0].split('-')[0]
    if name not in all_names:
        all_names.append(name)

print("Find {} images ...".format(len(all_names)))
lpips_scores = []
for name in tqdm(all_names):
    if "interfacegan" in args.img_dir:
        fpaths = [os.path.join(args.img_dir, '{}-{}.jpg'.format(name, idx)) for idx in range(args.num_samples)]
        images = []
        for path in fpaths:
            images.append(Variable(transform(Image.open(path).convert("RGB")).unsqueeze(0).cuda()))
        # compute the scores 
        fscores = []
        for i in range(args.num_samples//2-1):
            for j in range(i+1,args.num_samples//2):
                fscores.append(lpips(images[i], images[j]).cpu().item())
        mscores = []
        # compute the scores 
        for i in range(args.num_samples//2, args.num_samples-1):
            for j in range(i+1,args.num_samples):
                mscores.append(lpips(images[i], images[j]).cpu().item())
        lpips_scores.append([np.mean(fscores), np.mean(mscores)])
    else:
        # female
        fpaths = [os.path.join(args.img_dir, '{}-{}-0.jpg'.format(name, idx)) for idx in range(args.num_samples)]
        images = []
        for path in fpaths:
            images.append(Variable(transform(Image.open(path).convert("RGB")).unsqueeze(0).cuda()))
        fscores = []
        for i in range(args.num_samples-1):
            for j in range(i, args.num_samples):
                fscores.append(lpips(images[i], images[j]).cpu().item())
        # male
        fpaths = [os.path.join(args.img_dir, '{}-{}-1.jpg'.format(name, idx)) for idx in range(args.num_samples)]
        images = []
        for path in fpaths:
            images.append(Variable(transform(Image.open(path).convert("RGB")).unsqueeze(0).cuda()))
        mscores = []
        for i in range(args.num_samples-1):
            for j in range(i, args.num_samples):
                mscores.append(lpips(images[i], images[j]).cpu().item())
        lpips_scores.append([np.mean(fscores), np.mean(mscores)])

lpips_scores = np.array(lpips_scores)
print("Overall Average:", np.mean(lpips_scores))
print("Female Average:", np.mean(lpips_scores[:,0]))
print("Male Average:", np.mean(lpips_scores[:,1]))
