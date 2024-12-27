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


resized_image_size = 128

# Returns index for batches to
def _get_index_from_list(vals, t, x_shape):
    # x_shape of size (B, C, H, W)
    b_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(b_size, *((1, )  * (len(x_shape) - 1))).to(t.device) # new batch_x of size (B, 1, H, W)

# Face dataset with custom operations
class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, resized_image_size=128):
        self.dataset = dataset
        self.transform = transform
        self.resized_image_size = resized_image_size

    def _transform_image(self, image):
        # Define data transform
        transform = transforms.Compose([
            transforms.Resize((self.resized_image_size, self.resized_image_size)),
            transforms.ToTensor(), # Normalized betwen 0 and 1
            transforms.Lambda(lambda x: 2 * x - 1), # Scale bewteen [-1, 1]
        ])
        tensor = transform(image)
        return tensor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        if self.transform:
            image = self._transform_image(image)
        return image
    