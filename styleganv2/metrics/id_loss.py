
import numpy as np

import torch
from torch import nn
from models.encoders.model_irse import Backbone

class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        try:
            self.facenet.load_state_dict(torch.load('./pretrained_models/model_ir_se50.pth'))
        except IOError:
            self.facenet.load_state_dict(torch.load('/apdcephfs/share_916081/amosyhliu/pretrained_models/model_ir_se50.pth'))
        
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def cal_loss(self, y, y_hat):
        n_samples = y.size(0)
        y_feats = self.extract_feats(y)
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0.0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += (1 - diff_target)
        return loss/n_samples

    def cal_identity_similarity(self, y, y_hat):
        n_samples = y.size(0)
        y_feats = self.extract_feats(y)
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        similarities = []
        for i in range(n_samples):
            similarities.append(y_hat_feats[i].dot(y_feats[i]).cpu().item())
        return np.mean(similarities)
