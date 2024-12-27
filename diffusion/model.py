
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


# Constants
batch_size = 4
resized_size = 128
train_ratio = 0.8
timesteps = 300
dim_embeddings = 32 # Change to 32
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available else "cpu")

class PositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        pos_len = min(t.max(), timesteps)
        b = (torch.arange(pos_len + 1, device=device) / 10000.).unsqueeze(1) # (pos_len, 1)
        e = torch.arange(half_dim, device=device) / self.dim
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
    def __init__(self):
        super().__init__()
        self.image_channels = 3
        self.down_channels = (64, 128, 256, 512, 1024)
        self.up_channels = (1024, 512, 256, 128, 64)
        self.out_dim = 3
        self.time_emb_dim = 32
        self.time_mlp = nn.Sequential(
            PositionalEmbeddings(dim_embeddings),
            nn.Linear(dim_embeddings, dim_embeddings),
            nn.ReLU(),
        )

        self.conv0 = nn.Conv2d(self.image_channels, self.down_channels[0], 3, padding=1)
        
        self.downs = nn.ModuleList([
            Block(self.down_channels[i], self.down_channels[i + 1], self.time_emb_dim) for i in range(len(self.down_channels) - 1)
        ])
        self.ups = nn.ModuleList([
            Block(self.up_channels[i], self.up_channels[i + 1], self.time_emb_dim, up=True) for i in range(len(self.up_channels) - 1)
        ])

        self.output = nn.Conv2d(self.up_channels[-1], self.out_dim, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep) # (dim_embeddings)
        
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
