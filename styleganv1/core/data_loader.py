import os
import glob
import json
from munch import Munch
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.distributed as distributed

import random
random.seed(777)

attributes =[
    "Gender", 
    "Glasses", 
    "Age", 
    "Expression"
]

class ImageDataset(Dataset):
    def __init__(self, 
        source_path, 
        attr_path, 
        latent_path,
        transform=None,
        mode="train",
        latent_type='wp'
    ):
        """Initialize and preprocess the CelebAHQ dataset."""
        self.source_path   = source_path
        self.latent_path   = os.path.join(latent_path, mode, '{}.npy'.format(latent_type))
        self.attr_path     = os.path.join(attr_path, "list_attr_celeba-{}.txt".format(mode))
        self.transform     = transform
        self.selected_attrs = attributes
        self.mode          = mode
        self.num_selected_attrs = len(self.selected_attrs)

        all_images        =  glob.glob(os.path.join(source_path, mode, "*.jpg"))
        all_images.sort()
        self.all_images   = [f.split('/')[-1] for f in all_images]
        self.latent_codes = np.load(self.latent_path)

        self.image2attr   = {}
        self.attr2idx     = {}
        self.idx2attr     = {}
        self.preprocess(self.attr_path)

        self.num_images   = len(self.all_images)
        self.dataset_path = os.path.join(source_path, mode)
        print("Find {} {} images ...".format(len(self.all_images), mode))


    def preprocess(self, attr_path):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(int(values[idx] == '1'))

            self.image2attr[filename] = label

    def __getitem__(self, index):
        fname_src = self.all_images[index]
        label_src = self.image2attr[fname_src]
        image_src = Image.open(os.path.join(self.dataset_path, fname_src)).convert("RGB")
        latent    = self.latent_codes[index]

        fname_trg = random.choice(self.all_images)
        label_trg = self.image2attr[fname_trg]

        #label_rnd = (torch.rand(self.num_selected_attrs) > 0.5).float()
        label_src = torch.tensor(label_src).float()
        label_trg = torch.tensor(label_trg).float()
        latent    = torch.from_numpy(latent).float().squeeze(0)

        image_src = self.transform(image_src)
        inputs = [image_src, latent, label_src, label_trg]
        return inputs

    def __len__(self):
        return self.num_images


def get_data_loader(
    source_path,
    attr_path,
    latent_path,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    mode="train",
    latent_type='wp',
    is_distributed=False
):

    transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5])
    ])

    dataset = ImageDataset(
        source_path,
        attr_path,
        latent_path, 
        transform=transform, 
        mode=mode,
        latent_type=latent_type
    )

    if mode == "train":
        if is_distributed:
            num_tasks = distributed.get_world_size()
            global_rank = distributed.get_rank()
            sampler = torch.utils.data.DistributedSampler(
                dataset, 
                num_replicas=num_tasks, 
                rank=global_rank, 
                shuffle=True
            )
        else:
            sampler = None
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    return data_loader


class InputFetcher:
    def __init__(self, loader, device=None):
        self.loader = loader
        self.device = device

    def _fetch_inputs(self):
        try:
            inputs = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            inputs = next(self.iter)
        return inputs

    def __next__(self):
        data = self._fetch_inputs()
        inputs = Munch(
            img_src=data[0], 
            lat=data[1], 
            lab_src=data[2], 
            lab_trg=data[3]
        )
        return Munch({k: v.to(self.device) for k, v in inputs.items()})
