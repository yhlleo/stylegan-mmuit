
import os
import glob
import argparse
import copy
from munch import Munch
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
from torch.backends import cudnn
from torchvision import transforms
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as distributed

from core.solver import (
    run,
    build_stylegan_model
)
from core.model import MappingNetwork

from core.data_loader import (
    get_data_loader, 
    InputFetcher
)

def main(args):
    print(args)

    if args.mode == 'train':
        args.distributed = args.num_gpus > 1

        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
        else:
            rank = -1
            world_size = -1

        args.world_size = world_size
        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", 
                init_method="env://",
                world_size=world_size,
                rank=rank
            )
            torch.distributed.barrier()

            seed = 1234 + distributed.get_rank()
            torch.manual_seed(seed)
            np.random.seed(seed)
            cudnn.benchmark = True

            device = torch.device("cuda", distributed.get_rank())
        else:
            device = torch.device("cuda")
        
        run(args, device)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=4,
                        help='Number of domains')
    parser.add_argument('--rnd_dim', type=int, default=64,
                        help='Latent vector dimension')
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Latent code dimension of stylegan')
    parser.add_argument('--latent_type', type=str, default='wp', help='[z, w, wp]')
    parser.add_argument('--use_residual', type=int, default=0)
    parser.add_argument('--use_posweight', type=int, default=0)
    parser.add_argument('--norm_type', type=str, default='adaln', help='adaln | adain | none')
    parser.add_argument('--pre_norm', type=int, default=0)

    # weight for objective functions
    parser.add_argument('--lambda_r1', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cls', type=float, default=1,
                        help='Weight for classification')
    parser.add_argument('--lambda_lat_cyc', type=float, default=1,
                        help='Weight for latent cycle consistency loss')
    parser.add_argument('--lambda_src_cyc', type=float, default=1,
                        help='Weight for image reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=0.2,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--lambda_reid', type=float, default=0,
                        help='weight of face re-identification')
    parser.add_argument('--lambda_lpips', type=float, default=1,
                        help='weight of similarity between original image and generated image')
    parser.add_argument('--lambda_lpips_cyc', type=float, default=1,
                        help='weight of similarity between original image and generated image')
    parser.add_argument('--lambda_nb', type=float, default=0.1,
                        help='weight of neighbouring constraint')

    # training arguments
    parser.add_argument('--total_iters', type=int, default=100000,
                        help='Number of total iterations')
    parser.add_argument('--warmup_steps', type=int, default=5000,
                        help="Number of warmup steps")
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=8,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate for D, E and G')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10,
                        help='Number of generated images per domain during sampling')

    # misc
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', "test", "inter",
                        "inter_eval", "separation"],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')
    parser.add_argument('--dataset', type=str, default='stylegan-mmuit')
    parser.add_argument('--dist_mode', type=str, default='squared_l2', 
                        help='[l2 | squared_l2], the distance type of LPIPS')
    parser.add_argument('--gan_model_name', type=str, default="stylegan_celebahq")

    # directory for training
    parser.add_argument('--source_path', type=str)
    parser.add_argument('--attr_path', type=str)
    parser.add_argument('--latent_path', type=str)
    parser.add_argument('--sample_dir', type=str, default='expr/samples',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
                        help='Directory for saving network checkpoints')

    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr/eval',
                        help='Directory for saving metrics, i.e., FID and LPIPS')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='expr/results',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='assets/representative/celeba_hq/src',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                        help='output directory when aligning faces')
    parser.add_argument('--output_name', type=str)

    # step size
    parser.add_argument('--print_every', type=int, default=40)
    parser.add_argument('--sample_every', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--eval_every', type=int, default=100000)
    parser.add_argument('--save_dir', type=str, default='./')

    # distributed training
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    main(args)

