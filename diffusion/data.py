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

from utils import FaceDataset

class Data:

    def __init__(self, data_cfg: DictConfig, trainer_cfg: DictConfig):
        self.data_cfg = data_cfg
        self.trainer_cfg = trainer_cfg
        self.train_ratio = self.data_cfg.train_ratio
        self.truncate = self.data_cfg.truncate
        self.resized_image_size = self.data_cfg.resized_image_size

        self.batch_size = self.trainer_cfg.batch_size
        
    def transform(self, image):
        # Define data transform
        transform = transforms.Compose([
            transforms.Resize((self.resized_size, self.resized_size)),
            transforms.ToTensor(), # Normalized betwen 0 and 1
            transforms.Lambda(lambda x: 2 * x - 1), # Scale bewteen [-1, 1]
        ])
        tensor = transform(image)
        return tensor

    def reverse_transform(self, tensor):
        # Reverse the same data transform
        reverse_transform = transforms.Compose([
            transforms.Lambda(lambda x: ((x + 1) / 2)),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            transforms.Lambda(lambda x: x * 255.0),
            transforms.Lambda(lambda x: x.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])
        image = reverse_transform(tensor)
        return image
    
    def process_data(self, dataset_path='tonyassi/celebrity-1000'):
        # Define dataset
        dataset = load_dataset(dataset_path, split='train')
        face_dataset = FaceDataset(dataset, Data.transform, self.resized_image_size)

        dataset_size = len(face_dataset)
        if self.truncate != 1:
            truncate_ratio = self.truncate 
            
            truncated_size = int(dataset_size * truncate_ratio) 

            if truncated_size < dataset_size:
                face_dataset = face_dataset.select(range(truncated_size))

        truncated_dataset_size = len(face_dataset)
        train_size = int(truncated_dataset_size * self.train_ratio)
        test_size = truncated_dataset_size - train_size
        train_dataset, test_dataset = random_split(face_dataset, [train_size, test_size])

        # Load datasets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader
