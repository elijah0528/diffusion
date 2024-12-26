import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from datasets import load_dataset

import numpy as np
import matplotlib.pyplot as plt


from PIL import Image

# Constants
batch_size = 4
resized_size = 64
train_ratio = 0.8
timesteps = 20000

# Define data transform
transform = transforms.Compose([
    transforms.Resize((resized_size, resized_size)),
    transforms.ToTensor(), # Normalized betwen 0 and 1
    transforms.Lambda(lambda x: 2 * x - 1), # Scale bewteen [-1, 1]
])

# Reverse the same data transform
reverse_transform = transforms.Compose([
    transforms.Lambda(lambda x: ((x + 1) / 2)),
    transforms.Lambda(lambda x: x.permute(1, 2, 0)),
    transforms.Lambda(lambda x: x * 255.0),
    transforms.Lambda(lambda x: x.numpy().astype(np.uint8)),
    transforms.ToPILImage(),
])

# Convert a tensor to image and show it
def show_tensor_image(image):
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transform(image))

# Define dataset
dataset = load_dataset('tonyassi/celebrity-1000', split='train')

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        if self.transform:
            image = self.transform(image)
        return image


face_dataset = FaceDataset(dataset, transform)

# Get sizes to split train and test dataset
dataset_size = len(face_dataset)
train_size = int(dataset_size * train_ratio)
test_size = dataset_size - train_size
train_dataset, test_dataset = random_split(face_dataset, [train_size, test_size])

# Load datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

def get_index_from_list()

def forward_sample(x_0, t, device='cpu'):
    noise = torch.randn_like(x_0)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
    alpha_sqrt_cumprod_t = alpha_sqrt_cumprod[t]
    return alpha_sqrt_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise


# Betas is [steps]
betas = torch.linspace(0.0001, 0.02, steps=timesteps) # Scheduler from paper
alphas = 1 - betas # Creating a new variable for simplication
alpha_cumprod = torch.cumprod(alphas, axis=0) # For Forward diffusion thanks to reparameterization trick
alpha_sqrt_cumprod = torch.sqrt(alpha_cumprod) # For Forward diffusion thanks to reparameterization trick
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alpha_cumprod) # For Forward diffusion thanks to reparameterization trick

alpha_reci_sqrt = torch.sqrt(1 / alphas)  
alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)
posterior_variance = betas * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)

print(betas.shape)



noised_images = []

for batch in train_loader:
    print(batch.shape)
    for img in batch:
        print(img.max())
        for iter in range(timesteps):
            input = batch[0]
            if iter == 0:
                noised_images.append(input)
            # Size [C x H x W]
            noise = torch.randn_like(input)
            input = input + noise * betas[iter]
            if iter % 4000 == 0:
                noised_images.append(input)

        break

    
    num_images = min(5, len(noised_images))
    rows, cols = 1,5
    plt.figure(figsize=(2 * cols, 2 * rows))
    plt.axis('off')

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.title(f"iteration {i + 1}")
        show_tensor_image(noised_images[i])

    plt.tight_layout(pad=2.0)
    plt.show()
    # Pytorch images are C x H x W
    # Numpy images are H x W x C

    break
