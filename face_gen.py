import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt

class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_names = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

        if len(self.img_names) == 0:
            raise ValueError(f"Não foram encontradas imagens no diretório {root_dir}. Verifique se a pasta contém imagens válidas.")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.img_names[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

# Configurações gerais
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 100
lr = 0.0002
batch_size = 64
image_size = 64
epochs = 2

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset_path = '/home/douglasleodoro/data/celeba/img_align_celeba/img_align_celeba'
dataset = CelebADataset(root_dir=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img).view(-1, 1)

generator = Generator(z_dim).to(device)
discriminator = Discriminator().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()

# Treinamento com medidor de progresso
for epoch in range(epochs):
    total_batches = len(dataloader)
    for i, imgs in enumerate(dataloader):
        batch_size = imgs.size(0)

        real_imgs = imgs.to(device)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Discriminador
        z = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake_imgs = generator(z)

        real_loss = criterion(discriminator(real_imgs), real_labels)
        fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)
        loss_D = real_loss + fake_loss

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Gerador
        output = discriminator(fake_imgs)
        loss_G = criterion(output, real_labels)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # Progresso
        progress = (i + 1) / total_batches * 100
        print(f"\rEpoch {epoch+1}/{epochs} - Batch {i+1}/{total_batches} ({progress:.2f}%) - Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}", end='')

    print()  # Quebra linha após cada época

# Gerando imagens finais
generator.eval()
with torch.no_grad():
    z = torch.randn(16, z_dim, 1, 1).to(device)
    generated_imgs = generator(z).cpu()

grid = make_grid(generated_imgs, nrow=4, normalize=True)
plt.figure(figsize=(8, 8))
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.title("Amostras Geradas - Rostos (CelebA)")
plt.show()
