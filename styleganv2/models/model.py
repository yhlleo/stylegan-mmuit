import torch 
import torch.nn as nn

from .stylegan2.model import Generator

class StyleGAN2Generator(nn.Module):
    def __init__(
        self, 
        img_size=1024, 
        style_dim=512, 
        n_mlp=8, 
        input_is_latent=True, 
        randomize_noise=True
    ):
        super(StyleGAN2Generator, self).__init__()
        self.model = Generator(img_size, 512, 8)
        self.pool  = nn.AdaptiveAvgPool2d((256, 256))

        self.input_is_latent = input_is_latent
        self.randomize_noise = randomize_noise

    def _load_pretrain(self, path):
        ckpt = torch.load(path, map_location='cpu')['g_ema']
        self.model.load_state_dict(ckpt, strict=True)

    def forward_test(self, x):
        out = self.model(
            [x], 
            input_is_latent=self.input_is_latent, 
            randomize_noise=self.randomize_noise
        )[0]
        return out

    def forward(self, x):
        out = self.model(
            [x], 
            input_is_latent=self.input_is_latent, 
            randomize_noise=self.randomize_noise
        )[0]
         
        out = self.pool(out)
        return out