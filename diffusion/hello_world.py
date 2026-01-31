#!/usr/bin/env python3
"""
🎨 DDPM Forward Diffusion Demo (Hello World)

This script demonstrates the forward diffusion process - showing how
an image progressively becomes Gaussian noise over timesteps.

No trained model weights required! Just run:
    python hello_world.py

Based on: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
https://arxiv.org/abs/2006.11239
"""

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import os

# ============================================
# CONSTANTS (matching ddpm-config.yaml defaults)
# ============================================
TIMESTEPS = 1000
IMAGE_SIZE = 64
BETA_START = 0.0001
BETA_END = 0.02

# Timesteps to visualize
DEMO_TIMESTEPS = [0, 100, 200, 400, 600, 800, 999]


# ============================================
# DIFFUSION SCHEDULE SETUP
# ============================================
def setup_diffusion_schedule(timesteps=TIMESTEPS):
    """
    Set up the diffusion noise schedule.
    
    The key insight of DDPM is that we can jump to ANY timestep t directly
    using the cumulative product of alphas, rather than iterating step by step.
    
    Returns a dict with all the precomputed values we need.
    """
    # Linear beta schedule (from the paper)
    betas = torch.linspace(BETA_START, BETA_END, timesteps)
    
    # Alpha is just 1 - beta
    alphas = 1.0 - betas
    
    # Cumulative product of alphas - this is the magic!
    # alpha_cumprod[t] = alpha[0] * alpha[1] * ... * alpha[t]
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    
    # Precompute sqrt values for the forward diffusion formula
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alpha_cumprod': alpha_cumprod,
        'sqrt_alpha_cumprod': sqrt_alpha_cumprod,
        'sqrt_one_minus_alpha_cumprod': sqrt_one_minus_alpha_cumprod,
    }


# ============================================
# FORWARD DIFFUSION
# ============================================
def forward_diffuse(x_0, t, schedule):
    """
    Apply forward diffusion to image x_0 at timestep t.
    
    The forward diffusion formula (reparameterization trick):
        x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
    
    where:
        - x_0 is the original image
        - ε is random Gaussian noise
        - α̅_t is the cumulative product of alphas up to timestep t
    
    This lets us jump directly to any timestep without iterating!
    """
    # Sample random noise
    noise = torch.randn_like(x_0)
    
    # Get the precomputed values for timestep t
    sqrt_alpha_cumprod_t = schedule['sqrt_alpha_cumprod'][t]
    sqrt_one_minus_alpha_cumprod_t = schedule['sqrt_one_minus_alpha_cumprod'][t]
    
    # Apply the forward diffusion formula
    x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
    
    return x_t, noise


# ============================================
# IMAGE UTILITIES
# ============================================
def load_sample_image():
    """Load a single sample image from the celebrity dataset."""
    print("Loading sample celebrity image from HuggingFace...")
    
    # Stream just the first example to keep this demo fast
    dataset = load_dataset('tonyassi/celebrity-1000', split='train', streaming=True)
    example = next(iter(dataset))
    image = example['image']
    
    # Transform: resize, to tensor, normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1),  # Scale to [-1, 1]
    ])
    
    tensor = transform(image)
    print(f"✓ Loaded image ({IMAGE_SIZE}x{IMAGE_SIZE})")
    
    return tensor


def tensor_to_image(tensor):
    """Convert a [-1, 1] tensor back to a displayable image."""
    # Clone to avoid modifying original
    img = tensor.clone()
    
    # Scale from [-1, 1] to [0, 1]
    img = (img + 1) / 2
    
    # Clamp to valid range
    img = torch.clamp(img, 0, 1)
    
    # Convert to numpy and transpose to (H, W, C)
    img = img.permute(1, 2, 0).numpy()
    
    return img


def is_non_interactive_backend():
    """Return True if Matplotlib backend can't display windows."""
    backend = matplotlib.get_backend().lower()
    try:
        from matplotlib.backends import backend_registry, BackendFilter
        non_interactive = {
            bk.lower()
            for bk in backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE)
        }
    except Exception:
        non_interactive = {
            "agg",
            "cairo",
            "pdf",
            "pgf",
            "ps",
            "svg",
            "template",
        }
    return backend in non_interactive


# ============================================
# MAIN DEMO
# ============================================
def main():
    print("=" * 50)
    print("🎨 DDPM Forward Diffusion Demo")
    print("=" * 50)
    print()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Setup diffusion schedule
    print("Setting up diffusion schedule...")
    schedule = setup_diffusion_schedule()
    print(f"✓ Created linear beta schedule ({BETA_START} → {BETA_END})")
    print(f"✓ Timesteps: {TIMESTEPS}")
    print()
    
    # Load sample image
    x_0 = load_sample_image()
    print()
    
    # Apply forward diffusion at various timesteps
    print("Applying forward diffusion at various timesteps...")
    print()
    
    # Create figure for visualization
    n_images = len(DEMO_TIMESTEPS)
    fig, axes = plt.subplots(1, n_images, figsize=(2.5 * n_images, 3))
    
    for idx, t in enumerate(DEMO_TIMESTEPS):
        # Get noise level info
        alpha_cumprod_t = schedule['alpha_cumprod'][t].item()
        signal_ratio = alpha_cumprod_t * 100
        noise_ratio = (1 - alpha_cumprod_t) * 100
        
        if t == 0:
            # At t=0, just show original
            x_t = x_0
            print(f"  t={t:4d} → Original image (100% signal, 0% noise)")
        else:
            # Apply forward diffusion
            x_t, _ = forward_diffuse(x_0, t, schedule)
            print(f"  t={t:4d} → {signal_ratio:5.1f}% signal, {noise_ratio:5.1f}% noise")
        
        # Plot
        axes[idx].imshow(tensor_to_image(x_t))
        axes[idx].set_title(f't = {t}', fontsize=12)
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = os.path.join(os.path.dirname(__file__), 'images')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'hello_world_output.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print()
    print(f"✓ Saved visualization to: {output_path}")
    
    # Show the plot
    if is_non_interactive_backend():
        print(f"ℹ️  Skipping plt.show() (non-interactive backend: {matplotlib.get_backend()})")
    else:
        plt.show()
    
    # Educational summary
    print()
    print("=" * 50)
    print("📚 What just happened?")
    print("=" * 50)
    print("""
This demo showed the FORWARD diffusion process:
  • We started with a clear image (t=0)
  • Progressively added Gaussian noise
  • Ended with pure noise (t≈1000)

The math:  x_t = √(α̅_t) · x_0 + √(1-α̅_t) · ε

Where α̅_t decreases from 1→0 as t increases,
smoothly transitioning from image to noise.

🎯 TRAINING teaches a neural network to REVERSE this:
   Given x_t and t, predict the noise ε that was added.

🎨 GENERATION starts from pure noise and iteratively
   denoises to create a new image!
""")


if __name__ == "__main__":
    main()
