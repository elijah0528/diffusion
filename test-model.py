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


from PIL import Image, ExifTags

from main import SimpleUNet

# Constants
batch_size = 4
resized_size = 128
train_ratio = 0.8
timesteps = 300
dim_embeddings = 64 # Change to 32
max_epochs = 2
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available else "cpu")
print(device)

# Define data transform
transform = transforms.Compose([
    transforms.Resize((resized_size, resized_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # Normalized betwen 0 and 1
    transforms.Lambda(lambda x: 2 * x - 1), # Scale bewteen [-1, 1]
])

# Reverse the same data transform
reverse_transform = transforms.Compose([
    transforms.Lambda(lambda x: ((x + 1) / 2)),
    transforms.Lambda(lambda x: x.permute(1, 2, 0)),
    transforms.Lambda(lambda x: x * 255.0),
    transforms.Lambda(lambda x: torch.clamp(x, 0, 255)),
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


# Face dataset with custom operations
class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        if isinstance(image, Image.Image):
            # Optionally handle EXIF data here if needed
            # For example, you can skip EXIF processing
            # or handle orientation manually if required.
            pass
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
    out = vals.gather(-1, t.cpu())
    return out.reshape(b_size, *((1, )  * (len(x_shape) - 1))).to(t.device) # new batch_x of size (B, 1, H, W)


# Gets diffused sample at an arbitrary timestep
def forward_sample(x_0, t, device='cpu'):
    noise = torch.randn_like(x_0).to(device)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    alpha_sqrt_cumprod_t = get_index_from_list(alpha_sqrt_cumprod, t, x_0.shape)
    x_0 = x_0.to(device)
    return (alpha_sqrt_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)), noise.to(device)


# Betas is [steps]
betas = torch.linspace(0.0001, 0.02, steps=timesteps) # Scheduler from paper
alphas = 1 - betas # Creating a new variable for simplication
alpha_cumprod = torch.cumprod(alphas, axis=0) # For Forward diffusion thanks to reparameterization trick
alpha_sqrt_cumprod = torch.sqrt(alpha_cumprod) # For Forward diffusion thanks to reparameterization trick
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alpha_cumprod) # For Forward diffusion thanks to reparameterization trick


alpha_reci_sqrt = torch.sqrt(1 / alphas) # Used in sampling
alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0) # Used in sampling
posterior_variance = betas * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod) # Used in sampling

weights_path = 'diffusion-model-1.pth'

model = SimpleUNet().to(device)
model.load_state_dict(torch.load(weights_path))
model.eval()  # Set the model to evaluation mode

def load_and_test_model(weights_path, test_loader):
    # Initialize the model
    model = SimpleUNet().to(device)
    
    # Load the model weights
    model.load_state_dict(torch.load(weights_path))
    model.eval()  # Set the model to evaluation mode

    # Test the model on the test dataset
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            batch = batch.to(device)
            if len(batch) != batch_size:
                continue
            
            # Generate random timesteps for testing
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            x_noisy, _ = forward_sample(batch, t, device)  # Get noisy samples
            
            # Get the model's predictions
            noise_pred = model(x_noisy, t)
            
            # Visualize the results
            plt.figure(figsize=(15, 5))
            for i in range(batch_size):
                plt.subplot(1, batch_size, i + 1)
                show_tensor_image(noise_pred[i].detach().cpu())
                plt.savefig(f'output_image_{i}.png')  # Save the image
                plt.axis('off')
            plt.show()

            break  # Remove this if you want to test on the entire test_loader

# Call the function to load weights and test the model
# load_and_test_model('diffusion-model-1.pth', test_loader)  # Change the path as needed


@torch.no_grad()
def sample_timestep(x, t, model):
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
def sample_plot_image(model):

    img_size = resized_size
    img = torch.randn((1, 3, img_size, img_size)).to(device)
    plt.figure(figsize=(40,40))
    plt.axis('off')
    num_images = 10
    stepsize = int(timesteps/num_images)

    for i in range(0,timesteps)[::-1]:
        t = torch.full((1,), i, dtype=torch.long).to(device)
        img = sample_timestep(img, t, model)

        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
            plt.savefig(f'output_image_{i}.png')  # Save the image

    plt.savefig(f'output_image_final.png')  # Save the image


# Call the function to generate and save images
sample_plot_image(model)