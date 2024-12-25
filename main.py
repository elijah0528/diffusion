import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from datasets import load_dataset

import numpy
import matplotlib.pyplot as plt

from PIL import Image

""" Constants """
batch_size = 4
resized_size = 64
train_ratio = 0.8
noise_iterations = 175

dataset = load_dataset('tonyassi/celebrity-1000', split='train')

transform = transforms.Compose([
    transforms.Resize((resized_size, resized_size)),
    transforms.ToTensor(), # Normalized betwen 0 and 1
    transforms.Lambda(lambda x: 2 * x - 1), # Scale bewteen [-1, 1]
])

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
dataset_size = len(face_dataset)
train_size = int(dataset_size * train_ratio)
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(face_dataset, [train_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Betas is [steps]
betas = torch.linspace(0, 0.2, steps=noise_iterations)
print(betas.shape)


noised_images = []

for batch in train_loader:
    print(batch.shape)
    for img in batch:
        print(img.max())
        for iter in range(noise_iterations):
            input = batch[0]
            if iter == 0:
                noised_images.append(input)
            # Size [C x H x W]
            noise = torch.randn_like(input)
            input = input + noise * betas[iter]
            if iter % 8 == 0:
                noised_images.append(input)
            
        break

    
    num_images = len(noised_images)
    rows, cols = 5,5
    plt.figure(figsize=(16, 3 * rows))
    plt.axis('off')

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        numpy_img = noised_images[i].permute(1, 2, 0).numpy()
        plt.title(f"iteration {i + 1}")
        plt.imshow(numpy_img)
    plt.tight_layout(pad=2.0)
    plt.show()
    # Pytorch images are C x H x W
    # Numpy images are H x W x C

    break
