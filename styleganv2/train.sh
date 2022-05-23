#!/usr/bin/env bash
export http_proxy="9.131.211.13:3128"
export https_proxy="9.131.211.13:3128"

NUM_GPUS=4

DATASET=stylegan2-ffhq
GAN_BASE=ffhq # ffhq, cat2dog
LATENT_TYPE=wp

USE_RESIDUAL=0
PRE_NORM=1
TRAIN_TYPE=w0.01-ds2-lpips1-cls2-${LATENT_TYPE}-pn${PRE_NORM}

DATA_DIR=/path/to/ROOT_DIR 
DISK_DATA=${DATA_DIR}/datasets/${DATASET}
SAMPLE_DIR=${DATA_DIR}/stylegan-mmuit2/${DATASET}_samples_${TRAIN_TYPE}
CHECKPOINTS_DIR=${DATA_DIR}/stylegan-mmuit2/${DATASET}_checkpoints_${TRAIN_TYPE}
STYLEGAN2_PATH=${DATA_DIR}/pretrained_models/stylegan2-ffhq-config-f.pt

python3 -m torch.distributed.launch \
  --nproc_per_node=${NUM_GPUS} \
  --master_port=18001 train_mmuit.py \
  --mode train \
  --gan_base ${GAN_BASE} \
  --num_domains 4 \
  --batch_size 4 \
  --lambda_r1 1 \
  --lambda_lpips 1 \
  --lambda_lat_cyc 1 \
  --lambda_src_cyc 1 \
  --lambda_cls 2 \
  --lambda_reid 0 \
  --lambda_ds 2 \
  --lambda_nb 0.01 \
  --total_iters 50000 \
  --sample_every 500 \
  --warmup_steps 5000 \
  --eval_every 40000 \
  --save_every 5000 \
  --ds_iter 50000 \
  --source_path ${DISK_DATA} \
  --attr_path ${DISK_DATA} \
  --latent_path ${DISK_DATA} \
  --sample_dir ${SAMPLE_DIR} \
  --checkpoint_dir ${CHECKPOINTS_DIR} \
  --resume_iter 0 \
  --num_gpus ${NUM_GPUS} \
  --latent_type ${LATENT_TYPE} \
  --use_residual ${USE_RESIDUAL} \
  --use_posweight 1 \
  --norm_type adaln \
  --num_workers 2 \
  --pre_norm ${PRE_NORM} \
  --stylegan2_path ${STYLEGAN2_PATH} 