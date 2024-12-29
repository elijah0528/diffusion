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
import threading

import hydra
from omegaconf import DictConfig
from collections import defaultdict

import wandb
import random

from utils import _get_index_from_list, _show_tensor_image

import boto3
from botocore.exceptions import ClientError
import io
from dotenv import load_dotenv

load_dotenv()
torch.manual_seed(42)

def initialize_wandb(cfg: DictConfig):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="remote-diffusion-run-1",

        # track hyperparameters and run metadata
        config={"main_cfg": cfg}
    )

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

    def __init__(self, main_cfg: DictConfig, trainer_cfg: DictConfig, optim_cfg: DictConfig, compute_cfg: DictConfig, data_cfg: DictConfig, model):
        self.compute_cfg = compute_cfg
        if self.compute_cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available else "cpu")
        else:
            self.device = self.compute_cfg.device 
        initialize_wandb(main_cfg)

        self.trainer_cfg = trainer_cfg
        self.optim_cfg = optim_cfg
        self.data_cfg = data_cfg

        self.weight_decay = self.optim_cfg.weight_decay
        self.learning_rate = self.optim_cfg.learning_rate

        self.model = model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.callbacks = defaultdict(list)

        self.batch_size = self.trainer_cfg.batch_size
        self.timesteps = self.trainer_cfg.timesteps
        self.n_embed = self.trainer_cfg.n_embed
        self.max_epochs = self.trainer_cfg.max_epochs
        self.snapshot_path = self.trainer_cfg.snapshot_path
        self.checkpoint_interval = self.trainer_cfg.checkpoint_interval
        self.loss_estimation_context = self.trainer_cfg.loss_estimation_context
        self.load_weights = self.trainer_cfg.load_weights

        self.resized_image_size = self.data_cfg.resized_image_size

        Trainer.initialize(self.timesteps)

        self.train_loader = None
        self.test_loader = None
    def load_pretrained_weights(self):
        self.model.load_state_dict(torch.load(self.snapshot_path, weights_only=True))
        print("Loaded pretrained weights from", self.snapshot_path)
    def train(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader
        if os.path.exists(self.snapshot_path) and self.load_weights:
            self.load_pretrained_weights()
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
                print(f"Epoch: {iter + 1} / {self.max_epochs}, Step: {step} / {max_steps}, Loss: {loss.item()}")

                if step % self.checkpoint_interval == 0 and step != 0:
                    self.checkpoint(iter, step, max_steps, loss)
    
    # Gets diffused sample at an arbitrary timestep
    def forward_sample(self, x_0, t):
        noise = torch.randn_like(x_0).to(self.device)
        sqrt_one_minus_alphas_cumprod_t = _get_index_from_list(Trainer.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        alpha_sqrt_cumprod_t = _get_index_from_list(Trainer.alpha_sqrt_cumprod, t, x_0.shape)
        if x_0.dim == 3:
            x_0 = x_0.unsqueeze(0)
        if t.size(0) != x_0.size(0):
            print(t.shape, x_0.shape)

        # Everything in the return is sent to device
        return alpha_sqrt_cumprod_t.to(self.device) * x_0.to(self.device) + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device), noise.to(self.device)

    def get_loss(self, model, x_0, t):
        x_0 = x_0.to(self.device)
        t = t.to(self.device)
        x_noisy, noise = self.forward_sample(x_0, t) # x_noisy and x_noise are already on device
        noise_pred = model(x_noisy, t)
        return F.l1_loss(noise, noise_pred)

    def get_random_sample(self):
        # Convert loaders to lists for random sampling
        train_batch = next(iter(self.train_loader))
        test_batch = next(iter(self.test_loader))
        
        return train_batch.to(self.device), test_batch.to(self.device)


    def estimate_loss(self):
        train_losses = torch.zeros(self.loss_estimation_context).to(self.device)
        test_losses = torch.zeros(self.loss_estimation_context).to(self.device)
    
        self.model.eval()
        for i in range(self.loss_estimation_context):

            with torch.no_grad():
                t = torch.randint(0, self.timesteps, (self.batch_size, ), device=self.device).long()
                train_samples, test_samples = self.get_random_sample()

                if len(train_samples) != self.batch_size or len(test_samples) != self.batch_size:
                    continue
                
                train_loss = self.get_loss(self.model, train_samples, t)
                test_loss = self.get_loss(self.model, test_samples, t)
                train_losses[i] = train_loss
                test_losses[i] = test_loss
        self.model.train()
        return train_losses.mean(), test_losses.mean()


    def checkpoint(self, iter, step, max_steps, loss):
        train_loss, test_loss = self.estimate_loss()
        print(f"Epoch: {iter + 1} / {self.max_epochs}, Step: {step} / {max_steps}, Loss: {loss.item()}")
        print(f"Train loss: {train_loss :4f}, Test loss: {test_loss:4f}")
        print(f"Load weights: {self.load_weights}, learning_rate = {self.optimizer.param_groups[0]['lr']}")
        torch.save(self.model.state_dict(), self.snapshot_path)
        wandb.log({"Epoch": iter + 1, "Step": step, "Train loss": train_loss, "Test loss": test_loss})
        # wandb.save(self.snapshot_path) 
        def upload_to_s3():
            try:
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                    region_name='us-east-1'
                )
                bucket_name = 'runpoddiffusionbucket'
                s3_path = f'model_checkpoints/diffusion_model_weights.pth'
                
                s3_client.upload_file(self.snapshot_path, bucket_name, s3_path)
                print(f"Successfully uploaded model to s3://{bucket_name}/{s3_path}")
            except ClientError as e:
                print(f"Failed to upload to S3: {e}")
        # Start upload in background
        thread = threading.Thread(target=upload_to_s3)
        thread.start()   
            
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

        img_size = self.resized_image_size
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
