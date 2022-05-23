# ISF-GAN with StyleGAN v1


## Configuration

 - Python: 3.6.8
 - Pytorch: 1.5.1+cu101
 - torchvision: 0.6.1+cu101
 - tensorflow-gpu: 1.14.0
 - timm: 0.4.12
 - ffmpeg
 - munch 

## Download pretrained models:

Download pretrained models from [here](https://drive.google.com/drive/folders/1SK22PJ2Oq_2NcXNpU3aIpkZ2fnZfNCey?usp=sharing), and copy them into the foloder `models/pretrain`:

```
./models/pretrain
  |__karras2018iclr-celebahq-1024x1024.pkl
  |__karras2019stylegan-celebahq-1024x1024.pkl
  |__karras2019stylegan-ffhq-1024x1024.pkl
  |__pggan_celebahq.pth
  |__stylegan_celebahq.pth
  |__stylegan_ffhq.pth
```

## Preparing dataset

 - Generate data through stylegan or pggan:

```
# stylegan
$ python3 generate_data.py -m stylegan_ffhq -o data/stylegan_ffhq -n 80000
```

Similarly, collect 10000 testing images.

 - Download pretrained classifiers:

```
$ python3 preparing/download.py
```

 - Collect attributes of generated images:

```
$ python3 preparing/collect_attributes.py
$ python3 preparing/merge.py
```

Finally, we can organize the dataset in this format:

```
../datasets/stylegan-ffhq
  |__train
  |   |__xxx.jpg
  |   |__log.txt
  |   |__w.npy
  |   |__wp.npy
  |   |__z.npy
  |
  |__test
  |   |__xxx.jpg
  |   |__log.txt
  |   |__w.npy
  |   |__wp.npy
  |   |__z.npy
  |
  |__list_attr_ffhq-test.txt
  |__list_attr_ffhq-train.txt
```

## Training 

```
$ sh ./scripts/train.sh
```

## Testing

 - Image Editing

```
$ python3 isfgan_edit.py \
  --checkpoint_dir /path/to/checkpoint_dir \
  --use_post 1 \
  --save_dir /path/to/save_dir
```

 - Image interpolation

```
$ python3 isfgan_interp.py \
  --checkpoint_dir /path/to/checkpoint_dir \
  --use_post 1 \
  --save_dir /path/to/save_dir
```

 - Image sampling

```
$ python3 isfgan_sample.py \
  --checkpoint_dir /path/to/checkpoint_dir \
  --use_post 1 \
  --save_dir /path/to/save_dir
```

