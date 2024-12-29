import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from datasets import load_dataset

import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

import hydra
from omegaconf import DictConfig
from collections import defaultdict


import wandb
import random

from utils import _get_index_from_list, _show_tensor_image, reverse_transform_image
from model import SimpleUNet


class Tester:
    betas = None
    alphas = None
    alpha_cumprod = None
    alpha_sqrt_cumprod = None
    sqrt_one_minus_alphas_cumprod = None
    alpha_reci_sqrt = None
    alpha_cumprod_prev = None
    posterior_variance = None
    initialized = False

    @classmethod
    def initialize(cls, timesteps):
        if not cls.initialized:
            # Betas is [steps]
            cls.betas = torch.linspace(0.0001, 0.02, steps=timesteps) # Scheduler from paper
            cls.alphas = 1 - cls.betas # Creating a new variable for simplication
            cls.alpha_cumprod = torch.cumprod(cls.alphas, axis=0) # For Forward diffusion thanks to reparameterization trick
            cls.alpha_sqrt_cumprod = torch.sqrt(cls.alpha_cumprod) # For Forward diffusion thanks to reparameterization trick
            cls.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - cls.alpha_cumprod) # For Forward diffusion thanks to reparameterization trick
            
            cls.alpha_reci_sqrt = torch.sqrt(1 / cls.alphas) # Used in sampling
            cls.alpha_cumprod_prev = F.pad(cls.alpha_cumprod[:-1], (1, 0), value=1.0) # Used in sampling
            cls.posterior_variance = cls.betas * (1. - cls.alpha_cumprod_prev) / (1. - cls.alpha_cumprod) # Used in sampling
            
            cls.initialized = True

    def __init__(self, trainer_cfg: DictConfig, resized_image_size, device):
        self.save_images=True
        self.save_image_path="./images"
        self.image_counter = 0 

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available else "cpu")
        else:
            self.device = device
        self.snapshot_path = "test-model-weights.pth"
    
        self.trainer_cfg = trainer_cfg
        self.batch_size = self.trainer_cfg.batch_size
        self.timesteps = self.trainer_cfg.timesteps
        self.n_embed = self.trainer_cfg.n_embed
        self.max_epochs = self.trainer_cfg.max_epochs
        self.snapshot_path = self.trainer_cfg.snapshot_path
        self.checkpoint_interval = self.trainer_cfg.checkpoint_interval
        self.loss_estimation_context = self.trainer_cfg.loss_estimation_context
        self.load_weights = self.trainer_cfg.load_weights

        self.model = SimpleUNet(trainer_cfg).to(self.device)
        self.resized_image_size = resized_image_size
        
        Tester.initialize(self.timesteps)

        self.load_pretrained_weights()
    def load_pretrained_weights(self):
        self.model.load_state_dict(torch.load(self.snapshot_path, weights_only=True))
        print("Loaded pretrained weights from", self.snapshot_path)

    @torch.no_grad()
    def _sample_timestep(self, x, t):
        betas_t = _get_index_from_list(Tester.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = _get_index_from_list(Tester.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = _get_index_from_list(Tester.alpha_reci_sqrt, t, x.shape)
        
        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = _get_index_from_list(Tester.posterior_variance, t, x.shape)
        
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 
    @torch.no_grad()
    def sample_plot_image(self):
        img_size = self.resized_image_size
        img = torch.randn((1, 3, img_size, img_size)).to(self.device)
        plt.figure(figsize=(15,10))
        plt.axis('off')
        num_images = 10
        stepsize = int(self.timesteps/num_images)

        for i in range(0,self.timesteps)[::-1]:
            t = torch.full((1,), i, dtype=torch.long).to(self.device)
            img = self._sample_timestep(img, t)

            img = torch.clamp(img, -1.0, 1.0)
            if i % stepsize == 0:
                plt.subplot(1, num_images, int(i/stepsize)+1)
                _show_tensor_image(img.detach().cpu())
    
        if self.save_images:

            os.makedirs(self.save_image_path, exist_ok=True)
            plot_save_path = os.path.join(self.save_image_path, f"plot_{self.image_counter}.png")
            plt.savefig(plot_save_path)
            self.image_counter += 1
        plt.show()   


    
@hydra.main(config_path=".", config_name="ddpm-config", version_base="1.3")
def main(cfg: DictConfig):
    main_cfg = cfg
    resized_image_size = cfg["resized_image_size"]
    device = cfg["device"]
    data_cfg = cfg['data_cfg']
    trainer_cfg = cfg['trainer_cfg']
    optim_cfg = cfg['optimizer_cfg']
    compute_cfg = cfg['compute_cfg']
    

    t = Tester(trainer_cfg, resized_image_size, device)
    t.sample_plot_image()

if __name__ == "__main__":
    main()