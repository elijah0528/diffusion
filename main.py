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
max_epochs = 64

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

# Returns index for batches to
def get_index_from_list(vals, t, x_shape):
    # x_shape of size (B, C, H, W)
    b_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(b_size, *((1, )  * (len(x_shape) - 1)))# new batch_x of size (B, 1, H, W)


# Gets diffused sample at an arbitrary timestep
def forward_sample(x_0, t, device='cpu'):
    noise = torch.randn_like(x_0)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    alpha_sqrt_cumprod_t = get_index_from_list(alpha_sqrt_cumprod, t, x_0.shape)
    return alpha_sqrt_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise


# Betas is [steps]
betas = torch.linspace(0.0001, 0.02, steps=timesteps) # Scheduler from paper
alphas = 1 - betas # Creating a new variable for simplication
alpha_cumprod = torch.cumprod(alphas, axis=0) # For Forward diffusion thanks to reparameterization trick
alpha_sqrt_cumprod = torch.sqrt(alpha_cumprod) # For Forward diffusion thanks to reparameterization trick
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alpha_cumprod) # For Forward diffusion thanks to reparameterization trick

alpha_reci_sqrt = torch.sqrt(1 / alphas)  
alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)
posterior_variance = betas * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)



class PositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half_dim = self.dim // 2
        b = (torch.arange(100000) / 10000.).unsqueeze(1) # (pos_len, 1)
        e = torch.arange(half_dim) / self.dim
        e = e.unsqueeze(0) # (1, half_dim)
        embeddings = b ** e # (pos_len, half_dim)
        embeddings = torch.stack((embeddings.sin(), embeddings.cos()), dim=-1).flatten(start_dim=-2)
        # (100000, dim)
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

model = SimpleUNet()
optimizer = Adam(model.parameters(), lr=0.001)

def get_loss(model, x_0, t):
    x_noisy, noise = forward_sample(x_0, t)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(alpha_reci_sqrt, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:

        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 


@torch.no_grad()
def sample_plot_image():
    # Sample noise
    img_size = 128
    img = torch.randn((1, 3, img_size, img_size))
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(timesteps/num_images)

    for i in range(0,timesteps)[::-1]:
        t = torch.full((1,), i, dtype=torch.long)
        img = sample_timestep(img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    plt.show()   

noised_images = []
for iter in range(max_epochs):
    for step, batch in enumerate(train_loader):
        if step >= 100:  # Stop after 100 batches
            break
        optimizer.zero_grad()

        t = torch.randint(0, timesteps, (batch_size, )).long()
        loss = get_loss(model, batch, t)      
        loss.backward()
        optimizer.step()
        print(step, loss)
        if step % 5 == 0:
            print(f"Step {step} | step {step:03d} Loss: {loss.item()} ")
            sample_plot_image()

        # Pytorch images are C x H x W
        # Numpy images are H x W x C

    break
