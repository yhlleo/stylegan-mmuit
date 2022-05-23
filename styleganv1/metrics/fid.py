
import os
import argparse
import numpy as np
from scipy import linalg
from PIL import Image

import torch
import torch.nn as nn
from torch.utils import data
from torchvision import models
from torchvision import transforms

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)


class EvalDataset(data.Dataset):
    def __init__(self, im_list, transform=None):
        self.im_list = im_list
        self.transform = transform

    def __getitem__(self, index):
        fpath = self.im_list[index]
        im = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im

    def __len__(self):
        return len(self.im_list)

def get_eval_loader(im_list, img_size=256, batch_size=32):
    height, width = 299, 299
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    dataset = EvalDataset(im_list, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1,
                           pin_memory=True,
                           drop_last=False)


@torch.no_grad()
def calculate_fid_given_lists(im_lists, img_size=256, batch_size=32):
    print('Calculating FID given image lists ...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = InceptionV3().eval().to(device)
    loaders = [get_eval_loader(ll, img_size, batch_size) for ll in im_lists]

    mu, cov = [], []
    for loader in loaders:
        actvs = []
        for x in tqdm(loader, total=len(loader)):
            actv = inception(x.to(device))
            actvs.append(actv)
        actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
        mu.append(np.mean(actvs, axis=0))
        cov.append(np.cov(actvs, rowvar=False))
    fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
    return fid_value

@torch.no_grad()
def collect_fid(im_list, save_path, img_size=256, batch_size=32):
    print('Calculating FID given image lists ...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inception = InceptionV3().eval().to(device)
    loader = get_eval_loader(im_list, img_size, batch_size)

    actvs = []
    for x in tqdm(loader, total=len(loader)):
        actv = inception(x.to(device))
        actvs.append(actv)
    actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
    mu = np.mean(actvs, axis=0)
    cov = np.cov(actvs, rowvar=False)

    np.save(save_path, {"mu": mu, "cov": cov})


@torch.no_grad()
def calculate_fid_given_npy(im_list, npy_path, img_size=256, batch_size=32):
    print('Calculating FID given image lists ...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    src_param = np.load(npy_path, allow_pickle=True)
    mu1  = src_param.item().get("mu")
    cov1 = src_param.item().get("cov")

    inception = InceptionV3().eval().to(device)
    loader = get_eval_loader(im_list, img_size, batch_size)

    actvs = []
    for x in tqdm(loader, total=len(loader)):
        actv = inception(x.to(device))
        actvs.append(actv)
    actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
    mu2  = np.mean(actvs, axis=0)
    cov2 = np.cov(actvs, rowvar=False)

    fid_value = frechet_distance(mu1, cov1, mu2, cov2)
    return fid_value







