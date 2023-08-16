# Define VAE
import torch.nn.functional as F
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim, size = (128,128)):
        super(VAE, self).__init__()

        self.size = size
        
        self.fc1 = nn.Linear(self.size[0]*self.size[1], 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, self.size[0]*self.size[1])

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.size[0]*self.size[1]))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ConvVAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        
        # Latent vectors mu and logvar
        self.fc1 = nn.Linear(512*8*8, latent_dim)
        self.fc2 = nn.Linear(512*8*8, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 512*8*8)
        self.conv6 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv7 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv8 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv9 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv10 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)  # Flatten layer
        mu = self.fc1(x)
        logvar = self.fc2(x)
        return mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), 512, 8, 8)  # Unflatten/reshape layer
        z = F.relu(self.conv6(z))
        z = F.relu(self.conv7(z))
        z = F.relu(self.conv8(z))
        z = F.relu(self.conv9(z))
        z = torch.sigmoid(self.conv10(z))  # final layer should use sigmoid for values between 0-1
        return z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

