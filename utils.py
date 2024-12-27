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


# Returns index for batches to
def _get_index_from_list(vals, t, x_shape):
    # x_shape of size (B, C, H, W)
    b_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(b_size, *((1, )  * (len(x_shape) - 1))).to(t.device) # new batch_x of size (B, 1, H, W)

