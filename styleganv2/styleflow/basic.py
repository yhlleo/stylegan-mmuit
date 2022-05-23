import pickle
import numpy as np

def load_data(
    attr_path='styleflow_data/attributes.npy', 
    light_path='styleflow_data/light.npy', 
    sg2latents_path='styleflow_data/sg2latents.pickle'
):
    attributes = np.load(attr_path)
    light = np.load(light_path)
    latents = pickle.load(open(sg2latents_path, 'rb'))['Latent']
    return attributes, light, latents


def subset_selection(w, arr, index, value, model, zero_padding=None, use_selection=True):
    arr[:,9+index] = value
    rev = model(w, arr, zero_padding, True)
    if use_selection:
        if index == 0:
            rev[0][0][8:] = w[:, 8:]
        elif index == 1:
            rev[0][0][:2] = w[:, :2]
            rev[0][0][4:] = w[:, 4:]
        elif index == 2:
            rev[0][0][4:] = w[:, 4:]
        elif index == 3:
            rev[0][0][4:] = w[:, 4:]
        elif index == 4:
            rev[0][0][6:] = w[:, 6:]
        elif index == 5:
            rev[0][0][:5] = w[:, :5]
            rev[0][0][10:] = w[:, 10:]
        elif index == 6:
            rev[0][0][:4] = w[:, :4]
            rev[0][0][8:] = w[:, 8:]
        elif index == 7:
            rev[0][0][:4] = w[:, :4]
            rev[0][0][6:] = w[:, 6:]
    return rev