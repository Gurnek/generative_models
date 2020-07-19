import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import FashionMNIST, MNIST
from tqdm import tqdm

mnist = FashionMNIST('./data/', download=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    ]))

dataloader = DataLoader(mnist, batch_size=128, shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, 5, padding=1)
        self.conv2 = nn.Conv2d(4, 16, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(16, 32, 3) # 32 x 10 x 10
        self.pool2 = nn.MaxPool2d(2)

        self.lin1 = nn.Linear(800, 400)
        self.lin2 = nn.Linear(400, 200)
        self.lin3_1 = nn.Linear(200, 32)
        self.lin3_2 = nn.Linear(200, 32)

        self.lin4 = nn.Linear(32, 200)
        self.lin5 = nn.Linear(200, 400)
        self.lin6 = nn.Linear(400, 800)

        self.up1 = nn.Upsample(scale_factor=2)
        self.conv_t1 = nn.ConvTranspose2d(32, 16, 3)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv_t2 = nn.ConvTranspose2d(16, 4, 3)
        self.conv_t3 = nn.ConvTranspose2d(4, 1, 5, padding=1)

    def encode(self, x):
        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        x = self.pool1(x)
        x = F.selu(self.conv3(x))
        x = self.pool2(x)
        x = x.view(-1, 800)

        x = F.selu(self.lin1(x))
        x = F.selu(self.lin2(x))
        return self.lin3_1(x), self.lin3_2(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        z = F.selu(self.lin4(z))
        z = F.selu(self.lin5(z))
        z = F.selu(self.lin6(z))
        z = z.view(-1, 32, 5, 5)
        z = self.up1(z)
        z = F.selu(self.conv_t1(z))
        z = self.up2(z)
        z = F.selu(self.conv_t2(z))
        z = torch.sigmoid(self.conv_t3(z))
        return z

    def forward(self, x):
        #print(f'Input size is {x.shape}')
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        #print(f'Output size is {out.shape}')
        return out, mu, logvar

def vae_criterion(recon, x, mu, logvar):
    bce = F.binary_cross_entropy(recon.view(-1, 784), x.view(-1, 784), reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 5
    return bce + kld

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vae = VAE().to(device)
vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3, weight_decay=0)

epochs = 30


def vae_train():
    vae.train()

    total_loss = 0
    for data in tqdm(dataloader):
        data = data[0].to(device)
        vae_optimizer.zero_grad()
        recon, mu, logvar = vae(data)
        loss = vae_criterion(recon, data, mu, logvar)
        loss.backward()
        total_loss += loss.item()
        vae_optimizer.step()
    return total_loss

if __name__ == '__main__':
    print("TRAINING VAE")
    for e in range(epochs):
        loss = vae_train()
        print(f'Epoch {e}: loss {loss}')

    torch.save(vae, 'vae.pt')