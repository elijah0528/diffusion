## Diffusion Implementation
This repo implements a diffusion model trained on the CelebA dataset. The model architecture can be found here: https://arxiv.org/abs/2006.11239.

## Hello World: Forward Diffusion Demo
The quickest way to see diffusion in action is to run the forward noising demo:

```bash
cd diffusion
python3 hello_world.py
```

What it does:
- Downloads a single celebrity image from the `tonyassi/celebrity-1000` dataset
- Applies forward diffusion at multiple timesteps
- Saves a visualization to `diffusion/images/hello_world_output.png`

No trained weights are required for this demo.
