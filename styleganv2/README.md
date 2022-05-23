# ISF-GAN with StyleGAN v2


## Configuration

 - Python: 3.6.8
 - Pytorch: 1.6.0+cu101
 - torchvision: 0.7.0a0+78ed10c
 - timm: 0.4.5
 - ffmpeg
 - munch 

## Download pretrained models:

Download pretrained models from [here](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing).

## Preparing dataset

 - Generate data through stylegan2 (with the released data from StyleFlow):

```
# stylegan
$ python3 generate_data.py --save_dir /path/to/save_dir
```

 - Download pretrained classifiers:

```
$ python3 ../styleganv1/preparing/download.py
```

 - Collect attributes of generated images:

```
$ python3 ../styleganv1/preparing/collect_attributes.py
$ python3 ../styleganv1/preparing/merge.py
```

Finally, we can organize the dataset in this format:

```
../datasets/stylegan2-ffhq
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

