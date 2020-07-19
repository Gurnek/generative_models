import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import FashionMNIST, MNIST, CelebA
from torchvision.utils import save_image
from tqdm import tqdm

NUM_WORKERS = 2
BATCH_SIZE = 128
IMG_SIZE = 64
NUM_CHANNELS = 3
LATENT_SIZE = 100
NGF = 64
NDF = 64
NUM_EPOCHS = 5
LR = 2e-4
BETA1 = 0.5
NUM_GPU = 1

celeba = CelebA('./data/', download=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMG_SIZE),
    torchvision.transforms.CenterCrop(IMG_SIZE),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5])
]))

dataloader = DataLoader(celeba, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0., 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1., 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_SIZE, NGF*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(NGF*8, NGF*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(NGF*4, NGF*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(NGF*2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),

            nn.ConvTranspose2d(NGF, NUM_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, NGF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(NDF, NDF*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF*2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(NDF*2, NDF*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF*4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(NDF*4, NDF*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF*8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(NDF*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator = Generator().to(device)
discriminator = Discriminator().to(device)

if NUM_GPU > 1:
    generator = nn.DataParallel(generator, list(range(NUM_GPU)))
    discriminator = nn.DataParallel(discriminator, list(range(NUM_GPU)))

generator.apply(weights_init)
discriminator.apply(weights_init)

loss = nn.BCELoss().to(device)
optim_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))
optim_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))


noise_vecs = torch.randn(16, 32).to(device)

for epoch in range(NUM_EPOCHS):
    print(f'EPOCH {epoch}')
    g_error = 0
    d_error = 0
    for real_batch, _ in tqdm(dataloader):
        real_batch = real_batch.to(device)
        ones = torch.ones(real_batch.shape[0]).to(device)
        zeros = torch.zeros(real_batch.shape[0]).to(device)

        optim_g.zero_grad()
        z = torch.randn(real_batch.shape[0], LATENT_SIZE, 1, 1).to(device)
        fake_images = generator(z)
        error_g = loss(discriminator(fake_images).view(-1), ones)
        error_g.backward()
        g_error += error_g.item()
        optim_g.step()

        optim_d.zero_grad()
        error_real = loss(discriminator(real_batch).view(-1), ones)
        error_fake = loss(discriminator(fake_images.detach()).view(-1), zeros)
        error_d = (error_fake + error_real) / 2
        error_d.backward()
        d_error += error_d.item()
        optim_d.step()

    print(f'G ERROR: {g_error}')
    print(f'D ERROR: {d_error}')
    print()
    generated_vecs = generator(noise_vecs).detach()
    generated_imgs = generated_vecs.view(16, 3, 64, 64)
    save_image(generated_imgs, f'gan_images/{epoch}.png')