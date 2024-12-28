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
from trainer import Trainer
from data import Data
from utils import FaceDataset, _get_index_from_list

@hydra.main(config_path=".", config_name="ddpm-config", version_base="1.3")
def main(cfg: DictConfig):
    main_cfg = cfg
    data_cfg = cfg['data_cfg']
    trainer_cfg = cfg['trainer_cfg']
    optim_cfg = cfg['optimizer_cfg']
    compute_cfg = cfg['compute_cfg']
    
    model = SimpleUNet(trainer_cfg)

    d = Data(data_cfg, trainer_cfg)
    train_loader, test_loader = d.process_data()
    trainer = Trainer(main_cfg, trainer_cfg, optim_cfg, compute_cfg, data_cfg, model)
    trainer.train(train_loader, test_loader)
    print(trainer.batch_size)


if __name__ == "__main__":
    main()