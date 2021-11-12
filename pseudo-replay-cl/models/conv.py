import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Encoder(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Encoder, self).__init__()

        channels = img_shape[0] if img_shape[0] < img_shape[2] else img_shape[2]

        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
        )

        """
        self.linear_block = nn.Sequential(
            nn.Linear(128*14*14, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ELU(inplace=True),
        )
        """

        self.mu = nn.Linear(128*8*8, latent_dim)
        self.logvar = nn.Linear(128*8*8, latent_dim)
        self.latent_dim = latent_dim

    def reparameterization(self, mu, logvar,latent_dim):
        std = torch.exp(logvar / 2)
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
        z = sampled_z * std + mu
        return z
        
    def forward(self, img):
        x = self.conv_block(img)
        x = x.view(x.shape[0], -1)
        #x = self.linear_block(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterization(mu, logvar, self.latent_dim)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self,img_shape, latent_dim, n_classes, use_label=True):
        super(Decoder, self).__init__()

        channels = img_shape[0] if img_shape[0] < img_shape[2] else img_shape[2]

        # conditional generation
        input_dim = latent_dim + n_classes if use_label else latent_dim

        self.linear_block = nn.Sequential(
            #nn.Linear(input_dim, n_hidden),
            #nn.BatchNorm1d(n_hidden),
            #nn.ELU(inplace=True),
            #nn.Linear(n_hidden, 128*14*14),
            nn.Linear(input_dim, 128*8*8),
            nn.BatchNorm1d(128*8*8),
            nn.ELU(inplace=True),
        )

        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, channels, kernel_size=2, padding=0, stride=1),
            nn.Sigmoid(),
        )
        self.img_shape = img_shape

    def forward(self, z):
        x = self.linear_block(z)
        x = self.conv_block(x.view(-1, 128, 8, 8))
        return x


class Specific(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Specific, self).__init__()

        channels = img_shape[0] if img_shape[0] < img_shape[2] else img_shape[2]

        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
        )
        
        self.linear_block = nn.Sequential(
            nn.Linear(128*8*8, latent_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, img):
        x = self.conv_block(img)
        x = x.view(x.shape[0], -1)
        x = self.linear_block(x)
        return x
