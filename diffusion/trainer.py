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

from model import PositionalEmbeddings, Block, SimpleUNet
from utils import FaceDataset, _get_index_from_list

resized_size = 128

class Trainer:
    # Betas is [steps]
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

    def __init__(self, trainer_cfg: DictConfig, optim_cfg: DictConfig, compute_cfg: DictConfig, data_cfg: DictConfig, model):
        self.compute_cfg = compute_cfg
        if self.compute_cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available else "cpu")
        else:
            self.device = self.compute_cfg.device 

        self.trainer_cfg = trainer_cfg
        self.optim_cfg = optim_cfg
        self.data_cfg = data_cfg

        self.model = model.to(self.device)
        self.optimizer = None
        self.callbacks = defaultdict(list)

        self.batch_size = self.trainer_cfg.batch_size
        self.timesteps = self.trainer_cfg.timesteps
        self.dim_embeddings = self.trainer_cfg.dim_embeddings
        self.max_epochs = self.trainer_cfg.max_epochs
        self.snapshot_path = self.trainer_cfg.snapshot_path
        self.checkpoint_interval = self.trainer_cfg.checkpoint_interval

        self.weight_decay = self.optim_cfg.weight_decay
        self.learning_rate = self.optim_cfg.learning_rate

        self.resized_image_size = self.data_cfg.resized_image_size

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        Trainer.initialize(self.timesteps)

    def train(self, train_loader):
        max_steps = len(train_loader)
        for iter in range(self.max_epochs):
            for step, batch in enumerate(train_loader):
                if len(batch) != self.batch_size:
                    continue
                
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                t = torch.randint(0, self.timesteps, (self.batch_size, ), device=self.device).long()
                loss = self.get_loss(self.model, batch, t)      
                loss.backward()
                self.optimizer.step()
                print(step, loss.item())
                if step % self.checkpoint_interval == 0 and step != 0:
                    self.checkpoint(iter, step, max_steps, loss)
    
    # Gets diffused sample at an arbitrary timestep
    def forward_sample(self, x_0, t):
        noise = torch.randn_like(x_0).to(self.device)
        sqrt_one_minus_alphas_cumprod_t = _get_index_from_list(Trainer.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        alpha_sqrt_cumprod_t = _get_index_from_list(Trainer.alpha_sqrt_cumprod, t, x_0.shape)
        # Everything in the return is sent to device
        return alpha_sqrt_cumprod_t.to(self.device) * x_0.to(self.device) + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device), noise.to(self.device)

    def get_loss(self, model, x_0, t):
        x_0 = x_0.to(self.device)
        t = t.to(self.device)
        x_noisy, noise = self.forward_sample(x_0, t) # x_noisy and x_noise are already on device
        noise_pred = model(x_noisy, t)
        return F.l1_loss(noise, noise_pred)
            
    def checkpoint(self, iter, step, max_steps, loss):
        print(f"Epoch {iter + 1} / {self.max_epochs} | step {step} / {max_steps} Loss: {loss.item():4f} ")

    @torch.no_grad()
    def sample_timestep(self, x, t):
        betas_t = _get_index_from_list(Trainer.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = _get_index_from_list(
            Trainer.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = _get_index_from_list(Trainer.alpha_reci_sqrt, t, x.shape)
        
        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = _get_index_from_list(Trainer.posterior_variance, t, x.shape)
        
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 
    @torch.no_grad()
    def sample_plot_image(self):

        img_size = resized_size
        img = torch.randn((1, 3, img_size, img_size)).to(self.device)
        plt.figure(figsize=(15,15))
        plt.axis('off')
        num_images = 10
        stepsize = int(self.timesteps/num_images)

        for i in range(0,self.timesteps)[::-1]:
            t = torch.full((1,), i, dtype=torch.long).to(self.device)
            img = self.sample_timestep(img, t)

            img = torch.clamp(img, -1.0, 1.0)
            if i % stepsize == 0:
                plt.subplot(1, num_images, int(i/stepsize)+1)
                _show_tensor_image(img.detach().cpu())
        plt.show()   
