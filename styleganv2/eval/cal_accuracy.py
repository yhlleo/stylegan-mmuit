import sys
import pickle
import glob, os
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")
from dnnlib import tflib

classifier_urls = {
    0: "celebahq-classifier-00-male.pkl",
    1: "celebahq-classifier-01-smiling.pkl",
    2: "celebahq-classifier-04-young.pkl",
    3: "celebahq-classifier-12-black-hair.pkl",
    4: "celebahq-classifier-13-blond-hair.pkl",
    5: "celebahq-classifier-15-brown-hair.pkl",
    6: "celebahq-classifier-19-eyeglasses.pkl",
    7: "celebahq-classifier-34-straight-hair.pkl",
    8: "celebahq-classifier-03-wavy-hair.pkl",
    9: "celebahq-classifier-27-no-beard.pkl"
}

attributes_save_files = {
    0: "male.txt",
    1: "smiling.txt",
    2: "young.txt",
    3: "black-hair.txt",
    4: "blond-hair.txt",
    5: "brown-hair.txt",
    6: "eyeglasses.txt",
    7: "straight-hair.txt",
    8: "wavy-hair.txt",
    9: "no-beard.txt"
}

def load_pkl(file_or_url):
    with open(file_or_url, 'rb') as file:
        return pickle.load(file, encoding='latin1')


def main(_):
    val_attribute_list = [2, 0, 1, 6] # Age, Gender, Smile, Eyeglasses
    img_attribute_list = [6, 0, 7, 1] # Age, Gender, Smile, Eyeglasses
    attributes = ["Age", "Gender", "Smile", "Eyeglasses"]
    input_dir = './styleflow-results'

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    for idx, idy, attr in zip(val_attribute_list, img_attribute_list, attributes):
        model_path = classifier_urls[idx]
        all_imgs  = glob.glob(os.path.join(input_dir, '*-{}-*.jpg'.format(idy)))
        all_imgs.sort()
        num_imgs = len(all_imgs)

        correct = 0
        with tf.Session(config=config) as sess:
            # build classifier graph
            classifier = load_pkl(os.path.join("pretrained_models", model_path))
            images = tf.placeholder(tf.float32, [1, 3, 256, 256], name='inputs')
            logits = classifier.get_output_for(images, None)
            predictions = tf.sigmoid(logits)

            for fpath in tqdm(all_imgs):
                gt_lab = int(fpath.split('/')[-1].split('.')[0].split('-')[-1])
                img = np.asarray(Image.open(fpath).resize((256,256)), np.float32)[:,:,:3]
                img = np.transpose(img, [2,0,1])/255.0

                mu  = np.array([0.485, 0.456, 0.406])[:,np.newaxis,np.newaxis]
                std = np.array([0.229, 0.224, 0.225])[:,np.newaxis,np.newaxis]
                img = (img-mu)/std

                # note: the range of image is (0,1)
                pred = tflib.run(predictions, {images: img[np.newaxis,...]})
                pred_lab = 1 - int(pred[0][0] > 0.5) if attr != "Age" else int(pred[0][0] > 0.5)

                if gt_lab == pred_lab:
                    correct += 1
        print("{} ACC: {:.04f}".format(attr, float(correct)/num_imgs))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    tf.app.run(main)