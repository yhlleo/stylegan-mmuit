import os
import copy
import random
import time
import datetime
from munch import Munch
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed
from timm.scheduler.cosine_lr import CosineLRScheduler

from core.model import build_model
from core.data_loader import (
    InputFetcher,
    get_data_loader
)
from models.model_settings import MODEL_POOL
from models.pggan_generator2 import PGGANGenerator
from models.stylegan_generator2 import StyleGANGenerator
from utils.logger import setup_logger

import core.utils as utils
from metrics.lpips import LPIPS
from facenet import MTCNN, InceptionResnetV1

def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

def adv_multilabel_loss(logit, target, pos_weight=None):
    loss = F.binary_cross_entropy_with_logits(logit, target, reduction="mean", pos_weight=pos_weight)
    return loss

def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

def compute_d_loss(
    nets, 
    gan_model, 
    args, 
    img_src, 
    lat, 
    lab_src, 
    lab_trg, 
    pos_weight=None, 
    device=None
):
    all_losses = Munch()
    img_src.requires_grad_()

    lat = gan_model.preprocess(lat, latent_space_type=args.latent_type)
    if args.use_vae:
        rndz = torch.randn(img_src.size(0), 18 if args.latent_type == "wp" else 1, args.latent_dim).to(device).detach()
    else:
        rndz = torch.randn(img_src.size(0), args.rnd_dim).to(device).detach()
    lat_out = nets.mapnet(lat, rndz, lab_trg)
    lat_out = gan_model.preprocess(lat_out, latent_space_type=args.latent_type)
    img_trg = gan_model.synthesize(lat_out, latent_space_type=args.latent_type)["image"]
    img_trg = F.interpolate(img_trg, size=img_src.size()[2:], mode='bilinear') 

    real_cls, real_out = nets.discriminator(img_src)
    _, fake_out = nets.discriminator(img_trg)

    loss_cls = adv_multilabel_loss(real_cls, lab_src, pos_weight) * args.lambda_cls
    loss_r1  = r1_reg(real_cls, img_src) * args.lambda_r1

    loss_adv_real = adv_loss(real_out, 1)
    loss_adv_fake = adv_loss(fake_out, 0)

    all_losses = Munch(adv_real=loss_adv_real.item(),
                       adv_fake=loss_adv_fake.item(),
                       r1=loss_r1.item(),
                       cls=loss_cls.item())

    d_loss = loss_cls + loss_adv_real + loss_adv_fake + loss_r1
    return d_loss, all_losses

def compute_g_loss(
    nets, 
    gan_model, 
    args, 
    img_src, 
    lat, 
    lab_src, 
    lab_trg, 
    lpips=None, 
    mtcnn=None, 
    vggface=None, 
    pos_weight=None, 
    device=None
):
    
    lat = gan_model.preprocess(lat, latent_space_type=args.latent_type)
    if args.use_vae:
        rndz = torch.randn(img_src.size(0), 18 if args.latent_type == "wp" else 1, args.latent_dim).to(device).detach()
    else:
        rndz = torch.randn(img_src.size(0), args.rnd_dim).to(device).detach()
    lat_out = nets.mapnet(lat, rndz, lab_trg)
    lat_out = gan_model.preprocess(lat_out, latent_space_type=args.latent_type)
    img_trg = gan_model.synthesize(lat_out, latent_space_type=args.latent_type)["image"]
    img_trg = F.interpolate(img_trg, size=img_src.size()[2:], mode='bilinear') 

    # neighbouring constraint
    loss_nb = torch.mean(torch.norm(lat-lat_out, dim=(1, 2) if args.latent_type=="wp" else 1 )) * args.lambda_nb

    # adversarial loss
    fake_cls, fake_out = nets.discriminator(img_trg)
    loss_cls = adv_multilabel_loss(fake_cls, lab_trg, pos_weight) * args.lambda_cls
    loss_adv = adv_loss(fake_out, 1)

    # style reconstruction loss
    if args.use_vae:
        rndz = torch.randn(img_src.size(0), 18 if args.latent_type == "wp" else 1, args.latent_dim).to(device).detach()
    else:
        rndz = torch.randn(img_src.size(0), args.rnd_dim).to(device).detach()
    lat_cyc = nets.mapnet(lat_out, rndz, lab_src)
    lat_cyc = gan_model.preprocess(lat_cyc, latent_space_type=args.latent_type)
    loss_lat_cyc = torch.mean(torch.abs(lat_cyc - lat)) * args.lambda_lat_cyc

    # cycle reconstruction loss
    img_src_cyc = gan_model.synthesize(lat_cyc, latent_space_type=args.latent_type)["image"]
    img_src_cyc = F.interpolate(img_src_cyc, size=img_src.size()[2:], mode='bilinear') 
    loss_src_cyc = torch.mean(torch.abs(img_src_cyc - img_src)) * args.lambda_src_cyc

    # diversity sensitive 
    if args.use_vae:
        rndz = torch.randn(img_src.size(0), 18 if args.latent_type == "wp" else 1, args.latent_dim).to(device).detach()
    else:
        rndz = torch.randn(img_src.size(0), args.rnd_dim).to(device).detach()
    lat_out2 = nets.mapnet(lat, rndz, lab_trg)
    lat_out2 = gan_model.preprocess(lat_out2, latent_space_type=args.latent_type)
    img_trg2 = gan_model.synthesize(lat_out2, latent_space_type=args.latent_type)["image"]
    img_trg2 = F.interpolate(img_trg2, size=img_src.size()[2:], mode='bilinear') 
    loss_ds  = torch.mean(torch.abs(img_trg - img_trg2)) * args.lambda_ds

    all_losses = Munch(cls=loss_cls.item(),
                       adv=loss_adv.item(),
                       lat_cyc=loss_lat_cyc.item(),
                       src_cyc=loss_src_cyc.item(),
                       nb=loss_nb.item(),
                       ds=loss_ds.item())

    g_loss = loss_cls + \
             loss_adv + \
             loss_lat_cyc + \
             loss_src_cyc + \
             loss_nb - \
             loss_ds

    if args.lambda_lpips_cyc > 0:
        loss_lpips_cyc = lpips(img_src_cyc, img_src) * args.lambda_lpips_cyc
        all_losses.lpips_cyc = loss_lpips_cyc.item()
        g_loss += loss_lpips_cyc

    # content preservation loss
    if args.lambda_lpips > 0:
        loss_lpips = lpips(img_trg, img_src) * args.lambda_lpips
        all_losses.lpips = loss_lpips.item()
        g_loss += loss_lpips

    # id loss
    if args.lambda_reid > 0:
        img_src_cropped = mtcnn(img_src)
        img_trg_cropped = mtcnn(img_trg)
        if img_src_cropped is not None and img_trg_cropped is not None:
            loss_reid = vggface_dist(img_src_cropped, img_trg_cropped, vggface) * args.lambda_reid
            all_losses.reid = loss_reid.item()
            g_loss += loss_reid

    return g_loss, all_losses

def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def build_optims_schedulers(args, nets):
    optims = Munch()
    schedulers = Munch() 
    for net in nets.keys():
        optims[net] = torch.optim.Adam(
            nets[net].parameters(),
            lr=args.lr, betas=(args.beta1, args.beta2),
            weight_decay=1e-5
        )
        schedulers[net] = CosineLRScheduler(
            optims[net],
            t_initial=args.total_iters,
            lr_min=5e-6,
            warmup_lr_init=5e-7,
            warmup_t=args.warmup_steps,
            cycle_limit=1,
            t_in_epochs=False
        )
    return optims, schedulers


def load_checkpoint(model, checkpoint_dir, resume_iter=-1):
    if resume_iter > 0:
        ckpt = torch.load(os.path.join(checkpoint_dir, "{:06d}_nets_ema.ckpt".format(resume_iter)), map_location='cpu')
        
        for name, module in model.items():
            target_state_dict = module.state_dict()
            if name in ckpt:
                for key in target_state_dict:
                    if key in ckpt[name]:
                        target_state_dict[key] = ckpt[name][key]
                module.load_state_dict(target_state_dict)
        print('Loading checkpoint from %s...' % os.path.join(checkpoint_dir, "{:06d}_nets_ema.ckpt".format(resume_iter)))

def save_checkpoint(model, checkpoint_dir, step, suffix="nets"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    fname = os.path.join(checkpoint_dir, "{:06d}_{}.ckpt".format(step, suffix))
    print('Saving checkpoint into %s...' % fname)
    outdict = {}
    for name, module in model.items():
        outdict[name] = module.state_dict()
    torch.save(outdict, fname)

def reset_grad(optims):
    for optim in optims.values():
        optim.zero_grad()

def build_perceptual_nets(args, device):
    lpips = LPIPS(args.dist_mode, device).eval().to(device) if args.lambda_lpips > 0 or args.lambda_lpips_cyc>0 else None
    mtcnn = MTCNN().cuda().eval() if args.lambda_reid > 0 else None
    vggface = InceptionResnetV1(pretrained='vggface2').cuda().eval() if args.lambda_reid > 0 else None
    return lpips, mtcnn, vggface

def average_gradients(model):
    """ Gradient averaging. """
    size = float(distributed.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            distributed.all_reduce(param.grad.data, op=distributed.ReduceOp.SUM)
            param.grad.data /= size

def map_to_cuda(model, device):
    for _, module in model.items():
        module.to(device)

def build_stylegan_model(args):
    logger = setup_logger(args.checkpoint_dir, logger_name='load_stylegan')
    gan_type = MODEL_POOL[args.gan_model_name]['gan_type']
    if gan_type == 'pggan':
        model = PGGANGenerator(args.gan_model_name, logger)
    elif gan_type == 'stylegan':
        model = StyleGANGenerator(args.gan_model_name, logger)
    return model

def run(args, device=None):
    # build models
    nets, nets_ema = build_model(args, args.distributed, device)
    # print network details
    if (args.world_size==1 or distributed.get_rank() ==0):
        for name, module in nets.items():
            utils.print_network(module, name)

    # resume training if necessary
    load_checkpoint(nets, args.checkpoint_dir, args.resume_iter)
    if args.resume_iter > 0:
        nets_ema = copy.deepcopy(nets)
    # build optimizers and schedulers
    optims, schedulers = build_optims_schedulers(args, nets)

    # build perceptual nets
    lpips, mtcnn, vggface = build_perceptual_nets(args, device)
    # build stylegan 
    gan_model = build_stylegan_model(args)
    gan_model.model.eval().to(device)

    # build data loader
    loaders = Munch(src=get_data_loader(source_path=args.source_path,
                                        attr_path=args.attr_path, 
                                        latent_path=args.latent_path,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        mode="train",
                                        latent_type=args.latent_type,
                                        is_distributed=args.distributed),
                    val=get_data_loader(source_path=args.source_path,
                                        attr_path=args.attr_path, 
                                        latent_path=args.latent_path,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        mode="test",
                                        latent_type=args.latent_type,
                                        is_distributed=args.distributed))
    
    # fetch random validation images for debugging
    fetcher = InputFetcher(loaders.src, device=device)
    fetcher_val = InputFetcher(loaders.val, device=device)
    inputs_val = next(fetcher_val)

    pos_weight = None
    if args.use_posweight:
        pos_weight = torch.tensor([
            1.094125,  # Gender
            1.654650,  # Glasses
            1.718999,  # Age
            0.530550   # Expression
        ]).float().to(device)

    print('Start training...')
    start_time = time.time()
    for i in range(args.resume_iter, args.total_iters):
        # fetch images and labels
        inputs = next(fetcher)
        img_src, lat, lab_src, lab_trg = inputs.img_src, inputs.lat, inputs.lab_src, inputs.lab_trg

        # train discriminator 
        d_loss, d_losses = compute_d_loss(nets, gan_model, args, img_src, lat, lab_src, lab_trg,
            pos_weight, device)
        reset_grad(optims)
        d_loss.backward()
        #if args.world_size > 1:
        #    average_gradients(nets.discriminator)
        torch.nn.utils.clip_grad_norm_(nets.discriminator.parameters(), 2.0)
        optims.discriminator.step()

        # train the mapnet
        g_loss, g_losses = compute_g_loss(nets, gan_model, args, img_src, lat, lab_src, lab_trg, 
        	lpips=lpips, mtcnn=mtcnn, vggface=vggface, pos_weight=pos_weight, device=device)
        reset_grad(optims)
        g_loss.backward()
        #if args.world_size > 1:
        #    average_gradients(nets.mapnet)
        torch.nn.utils.clip_grad_norm_(nets.mapnet.parameters(), 2.0)
        optims.mapnet.step()

        # update learning rate
        schedulers.mapnet.step_update(i)
        schedulers.discriminator.step_update(i)

        # compute moving average of network parameters
        if args.distributed:
            moving_average(nets.mapnet.module, nets_ema.mapnet, beta=0.999)
            moving_average(nets.discriminator.module, nets_ema.discriminator, beta=0.999)
        else:
            moving_average(nets.mapnet, nets_ema.mapnet, beta=0.999)
            moving_average(nets.discriminator, nets_ema.discriminator, beta=0.999)

        # save model checkpoints
        if (args.world_size==1 or distributed.get_rank() ==0) and (i+1) % args.save_every == 0:
            save_checkpoint(nets_ema, args.checkpoint_dir, i+1, "nets_ema")
            save_checkpoint(optims, args.checkpoint_dir, i+1, "optims")

        # print out log info
        if (args.world_size==1 or distributed.get_rank() ==0) and (i+1) % args.print_every == 0:
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
            all_losses = dict()
            for loss, prefix in zip([d_losses, g_losses], ['D/loss_', 'G/loss_']):
                for key, value in loss.items():
                    all_losses[prefix + key] = value
            
            for opt in optims.keys():
                all_losses["lr/{}".format(opt)] = optims[opt].param_groups[0]['lr']
            log += ' '.join(['%s: [%.4f]' % (key, value) if 'lr' not in key else '%s: [%.6f]' % (key, value) for key, value in all_losses.items()])
            print(log)

        # generate images for debugging
        if (args.world_size==1 or distributed.get_rank() ==0) and (i+1) % args.sample_every == 0:
            os.makedirs(args.sample_dir, exist_ok=True)
            for net in nets_ema.keys():
                nets_ema[net].eval()
            with torch.no_grad():
                utils.debug_image(nets_ema, gan_model, args, inputs=inputs_val, step=i+1, device=device)

