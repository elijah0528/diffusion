
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

class PositionalEmbeddings(nn.Module):
    def __init__(self, n_embed, timesteps):
        super().__init__()
        self.n_embed = n_embed
        self.timesteps = timesteps
    def forward(self, t):
        device = t.device
        half_dim = self.n_embed // 2
        pos_len = min(t.max(), self.timesteps)
        b = (torch.arange(pos_len + 1, device=device) / 10000.).unsqueeze(1) # (pos_len, 1)
        e = torch.arange(half_dim, device=device) / self.n_embed
        e = e.unsqueeze(0) # (1, half_dim)
        embeddings = b ** e # (pos_len, half_dim)
        embeddings = torch.stack((embeddings.sin(), embeddings.cos()), dim=-1).flatten(start_dim=-2)
        embeddings = embeddings[t] 
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        self.up = up

        if up:
            self.conv1 = nn.Conv2d(2 * self.in_ch, self.out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(self.out_ch, self.out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(self.in_ch, self.out_ch, 3, padding=1)
            self.transform = nn.Conv2d(self.out_ch, self.out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))

        time_emb = self.relu(self.time_mlp(t)) # Time embedding is of shape (out_ch)
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb

        h = self.bnorm2(self.relu(self.conv2(h)))

        return self.transform(h)
    

class SimpleUNet(nn.Module):
    def __init__(self, trainer_cfg: DictConfig):
        super().__init__()
        self.trainer_cfg = trainer_cfg
        self.n_embed = self.trainer_cfg.n_embed
        self.timesteps = self.trainer_cfg.timesteps


        self.image_channels = 3
        self.down_channels = (64, 128, 256, 512, 1024)
        self.up_channels = (1024, 512, 256, 128, 64)

        self.time_mlp = nn.Sequential(
            PositionalEmbeddings(self.n_embed, self.timesteps),
            nn.Linear(self.n_embed, self.n_embed),
            nn.ReLU(),
        )

        self.conv0 = nn.Conv2d(self.image_channels, self.down_channels[0], 3, padding=1)
        
        self.downs = nn.ModuleList([
            Block(self.down_channels[i], self.down_channels[i + 1], self.n_embed) for i in range(len(self.down_channels) - 1)
        ])
        self.ups = nn.ModuleList([
            Block(self.up_channels[i], self.up_channels[i + 1], self.n_embed, up=True) for i in range(len(self.up_channels) - 1)
        ])

        self.output = nn.Conv2d(self.up_channels[-1], self.image_channels, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep) # (n_embed)
        
        x = self.conv0(x)

        residuals = []
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)

        for up in self.ups:
            residual_x = residuals.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        
        return self.output(x)