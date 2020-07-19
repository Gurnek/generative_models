import torch
from vae import VAE
from torchvision.utils import save_image

model = torch.load('model.pt')

with torch.no_grad():
    sample = torch.randn(32, 32).to('cuda')
    decoded_normal = model.decode(sample)
    save_image(decoded_normal.view(32, 1, 28, 28), 'normal.png')

    noise = torch.randn(32, 32).to('cuda') * 0.3
    decoded_noisy = model.decode(sample + noise)
    save_image(decoded_noisy.view(32, 1, 28, 28), 'noisy.png')