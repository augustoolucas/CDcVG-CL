import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def reparameterization(mu, logvar,latent_dim):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    z = sampled_z * std + mu
    return z

class Encoder(nn.Module):
    def __init__(self, img_shape, n_hidden, latent_dim):
        super(Encoder, self).__init__()

        channels = img_shape[0] if img_shape[0] < img_shape[2] else img_shape[2]

        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        """
        self.linear_block = nn.Sequential(
            nn.Linear(128*14*14, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ELU(inplace=True),
        )
        """

        self.mu = nn.Linear(128*14*14, latent_dim)
        self.logvar = nn.Linear(128*14*14, latent_dim)
        self.latent_dim = latent_dim
        
    def forward(self, img):
        x = self.conv_block(img)
        x = x.view(x.shape[0])
        #x = self.linear_block(x, -1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar, self.latent_dim)
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self,img_shape, n_hidden, latent_dim, n_classes, use_label=True):
        super(Decoder, self).__init__()

        channels = img_shape[0] if img_shape[0] < img_shape[2] else img_shape[2]

        # conditional generation
        input_dim = latent_dim + n_classes if use_label else latent_dim

        self.linear_block = nn.Sequential(
            #nn.Linear(input_dim, n_hidden),
            #nn.BatchNorm1d(n_hidden),
            #nn.ELU(inplace=True),
            #nn.Linear(n_hidden, 128*14*14),
            nn.Linear(input_dim, 128*14*14),
            nn.BatchNorm1d(128*14*14),
            nn.ReLU(inplace=True),
        )

        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, channels, kernel_size=2, padding=0, stride=1),
            nn.Sigmoid(),
        )
        self.img_shape = img_shape

    def forward(self, z):
        x = self.linear_block(z)
        x = self.conv_block(x.view(-1, 128, 14, 14))
        return x


class Specific(nn.Module):
    def __init__(self, img_shape, specific_n_hidden):
        super(Specific, self).__init__()

        channels = img_shape[0] if img_shape[0] < img_shape[2] else img_shape[2]
        # specific module
        self.specific = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), specific_n_hidden),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, imgs):
        x = self.specific(imgs.view(imgs.shape[0], -1))
        return x
     
class Classifier(nn.Module):
    def __init__(self, invariant_n_hidden, specific_n_hidden, classification_n_hidden, n_classes):
        super(Classifier, self).__init__()

        # classification module
        self.classifier_layer = nn.Sequential(
            nn.Linear(specific_n_hidden + invariant_n_hidden, classification_n_hidden),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Sequential(
            nn.Linear(classification_n_hidden, n_classes),
        )

    def forward(self, discriminative, invariant):
        x = self.classifier_layer(torch.cat([discriminative, invariant], dim=1))
        logits = self.output(x)
        return logits
