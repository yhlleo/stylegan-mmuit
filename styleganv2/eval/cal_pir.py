
import os
import sys
import glob
import argparse 
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow as tf

import torch
from torchvision import transforms
from torch.autograd import Variable

sys.path.append(".")
sys.path.append("..")
import dnnlib
from dnnlib import tflib
from metrics.coeff_var import coefficient_variation_maxmin
from metrics.lpips import LPIPS

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str)
parser.add_argument('--num_interp', type=int, default=20)
parser.add_argument('--model_type', type=str, default='th', help='th | tf')
args = parser.parse_args()

def open_file_or_url(file_or_url):
    if dnnlib.util.is_url(file_or_url):
        return dnnlib.util.open_url(file_or_url, cache_dir='.stylegan-cache')
    return open(file_or_url, 'rb')

def load_pkl(file_or_url):
    with open_file_or_url(file_or_url) as file:
        return pickle.load(file, encoding='latin1')

def tf_img_load(path, img_size=256):
    image = np.asarray(Image.open(path).resize((img_size,img_size)), np.float32)[:,:,:3]
    image = np.transpose(image, [2,0,1])
    return image

all_imgs = glob.glob(os.path.join(args.img_dir, '*-*.jpg'))
all_imgs.sort()
all_names = []
for fpath in all_imgs:
    name = fpath.split('/')[-1].split('.')[0].split('-')[0]
    if name not in all_names:
        all_names.append(name)

print("Find {} images ...".format(len(all_names)))

pir_scores = []
ppl_scores = []
if args.model_type == "tf":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        img1 = tf.placeholder(tf.float32, [1, 3, 256, 256], name='inputs')
        img2 = tf.placeholder(tf.float32, [1, 3, 256, 256], name='inputs')
        vgg_model = load_pkl('../checkpoints/vgg16_zhang_perceptual.pkl')
        score = vgg_model.get_output_for(img1, img2)

        for name in tqdm(all_names):
            fpaths = [os.path.join(args.img_dir, '{}-{}.jpg'.format(name, idx)) for idx in range(args.num_interp)]

            scores = []
            for i in range(args.num_interp - 1):
                image1 = tf_img_load(fpaths[i])
                image2 = tf_img_load(fpaths[i+1])
                pred = tflib.run(score, {img1: image1[np.newaxis,...], img2: image2[np.newaxis,...]}) / ((1.0/args.num_interp)**2)
                scores.append(pred)
                
            scores = np.array(scores).reshape(-1, args.num_interp-1)
            ppl_scores.append(np.mean(scores))
                
            image1 = tf_img_load(fpaths[0])
            image2 = tf_img_load(fpaths[-1])
            shortest_path = tflib.run(score, {img1: image1[np.newaxis,...], img2: image2[np.newaxis,...]})
            pir_scores.append(coefficient_variation_maxmin(scores, shortest_path))
else:
    device = torch.device('cuda')
    lpips = LPIPS(device=device).eval().cuda()

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])])
    for im in tqdm(all_names):
        # load images in sequence 
        fpaths = [os.path.join(args.img_dir, '{}-{}.jpg'.format(name, idx)) for idx in range(args.num_interp)]
            
        images = []
        for fp in fpaths:
            images.append(Variable(transform(Image.open(fp).convert('RGB')).unsqueeze(0).to(device)))

        scores = []
        for i in range(args.num_interp - 1):
            scores.append(lpips(images[i], images[i+1]).cpu().item())
        shortest_path = lpips(images[0], images[-1]).cpu().item() 

        scores = np.array(scores).reshape(-1, args.num_interp-1)
        ppl_scores.append(np.mean(scores))
        pir_scores.append(coefficient_variation_maxmin(scores, shortest_path))

pir_scores = np.array(pir_scores)
print("PIR score:", np.mean(pir_scores))
ppl_scores = np.array(ppl_scores)
print("PPL score:", np.mean(ppl_scores))
