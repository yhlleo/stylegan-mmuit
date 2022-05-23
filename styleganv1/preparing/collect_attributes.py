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
    3: "celebahq-classifier-19-eyeglasses.pkl"
}

attributes_save_files = {
    0: "male.txt",
    1: "smiling.txt",
    2: "young.txt",
    3: "eyeglasses.txt"
}

def load_pkl(file_or_url):
    with open(file_or_url, 'rb') as file:
        return pickle.load(file, encoding='latin1')

def main(_):
    input_dir = './data/stylegan_ffhq'
    save_folder = './attributes/train' # train or test
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    input_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
    input_paths.sort()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    
    for idx in [0,1,2,3,4,5,6,7,8,9]:
        pkl_path  = classifier_urls[idx]
        save_path = attributes_save_files[idx]

        with tf.Session(config=config) as sess, \
            open(os.path.join(save_folder, save_path), "w") as fout:
            # build classifier graph
            classifier = load_pkl(os.path.join("pretrained_models", pkl_path))
            images = tf.placeholder(tf.float32, [1, 3, 256, 256], name='inputs')
            logits = classifier.get_output_for(images, None)
            predictions = tf.sigmoid(logits)

            # output results
            for input_path in tqdm(input_paths):
                img = np.asarray(Image.open(input_path).resize((256,256)), np.float32)[:,:,:3]
                img = np.transpose(img, [2,0,1])/255.0

                mu  = np.array([0.485, 0.456, 0.406])[:,np.newaxis,np.newaxis]
                std = np.array([0.229, 0.224, 0.225])[:,np.newaxis,np.newaxis]
                img = (img-mu)/std

                # note: the range of image is (0,1)
                pred = tflib.run(predictions, {images: img[np.newaxis,...]})
                fname = input_path.split('/')[-1]
                fout.write("{}\t{:.04f}\n".format(fname, pred[0][0]))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    tf.app.run(main)

