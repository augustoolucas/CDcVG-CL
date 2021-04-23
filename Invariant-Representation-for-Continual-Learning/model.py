# Author: Ghada Sokar et al.
# This is the implementation for the Learning Invariant Representation for Continual Learning paper in AAAI workshop on Meta-Learning for Computer Vision

import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reparameterization(mu, logvar,latent_dim):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    z = sampled_z * std + mu
    return z


class ConvEncoder(nn.Module):
    def __init__(self, num_latent, img_channels):
        super(ConvEncoder, self).__init__()

        self.enc_conv_1 = nn.Conv2d(in_channels=img_channels,
                                    out_channels=16,
                                    kernel_size=(6, 6),
                                    stride=(2, 2))

        self.enc_conv_2 = nn.Conv2d(in_channels=16,
                                    out_channels=32,
                                    kernel_size=(4, 4),
                                    stride=(2, 2))

        self.enc_conv_3 = nn.Conv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=(2, 2),
                                    stride=(2, 2))

        self.z_mean = nn.Linear(64*2*2, num_latent)
        self.z_log_var = nn.Linear(64*2*2, num_latent)

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(device)
        z = z_mu + eps * torch.exp(z_log_var/2.).to(device)
        return z

    def encoder(self, img):
        x = self.enc_conv_1(img)
        x = F.leaky_relu(x)
        #print('conv1 out:', x.size())

        x = self.enc_conv_2(x)
        x = F.leaky_relu(x)
        #print('conv2 out:', x.size())

        x = self.enc_conv_3(x)
        x = F.leaky_relu(x)
        #print('conv3 out:', x.size())

        z_mean = self.z_mean(x.view(-1, np.prod(x.size()[1:])))
        z_log_var = self.z_log_var(x.view(-1, np.prod(x.size()[1:])))
        encoded = self.reparameterize(z_mean, z_log_var)
        #print('reparameterize out:', encoded.size())

        return encoded, z_mean, z_log_var

    def forward(self, img):
        return self.encoder(img)


class ConvDecoder(nn.Module):
    def __init__(self, num_latent, num_classes, img_channels):
        super(ConvDecoder, self).__init__()
        self.dec_linear_1 = nn.Linear(num_latent+num_classes, 64*2*2)

        self.dec_deconv_1 = nn.ConvTranspose2d(in_channels=64,
                                               out_channels=32,
                                               kernel_size=(2, 2),
                                               stride=(2, 2),
                                               padding=0)

        self.dec_deconv_2 = nn.ConvTranspose2d(in_channels=32,
                                               out_channels=16,
                                               kernel_size=(4, 4),
                                               stride=(3, 3),
                                               padding=1)
                                               #output_padding=1)

        self.dec_deconv_3 = nn.ConvTranspose2d(in_channels=16,
                                               out_channels=img_channels,
                                               kernel_size=(6, 6),
                                               stride=(3, 3),
                                               padding=4)

    def decoder(self, encoded):
        x = self.dec_linear_1(encoded)
        x = x.view(-1, 64, 2, 2)
        #print('dec_linear_1 out:', x.size())

        x = self.dec_deconv_1(x)
        x = F.leaky_relu(x)
        #print('deconv1 out:', x.size())

        x = self.dec_deconv_2(x)
        x = F.leaky_relu(x)
        #print('deconv2 out:', x.size())

        x = self.dec_deconv_3(x)
        x = F.leaky_relu(x)
        #print('deconv1 out:', x.size())

        decoded = torch.sigmoid(x)
        return decoded

    def forward(self, encoded):
        return self.decoder(encoded)


class Encoder(nn.Module):
    def __init__(self, img_shape, n_hidden, latent_dim, enlarged=False):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(nn.Linear(int(np.prod(img_shape)), n_hidden),
                                   nn.BatchNorm1d(n_hidden),
                                   nn.ReLU(inplace=True))

        enlarged_modules = nn.Sequential(nn.Linear(n_hidden, int(n_hidden//2)),
                                         nn.BatchNorm1d(int(n_hidden//2)),
                                         nn.ReLU(inplace=True))

        reparameterization_input_size = n_hidden

        if enlarged:
            self.model.add_module('enlarged_modules', enlarged_modules)
            reparameterization_input_size = int(n_hidden//2)

        self.mu = nn.Linear(reparameterization_input_size, latent_dim)
        self.logvar = nn.Linear(reparameterization_input_size, latent_dim)
        self.latent_dim = latent_dim
        
    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar, self.latent_dim)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self,
                 img_shape,
                 n_hidden,
                 latent_dim,
                 n_classes,
                 use_label=True,
                 enlarged=False):
        super(Decoder, self).__init__()
        # conditional generation
        if use_label:
            input_dim = latent_dim + n_classes
        else:
            input_dim = latent_dim

        if enlarged:
            self.model = nn.Sequential(nn.Linear(input_dim, int(n_hidden//2)),
                                       nn.BatchNorm1d(int(n_hidden//2)),
                                       nn.Linear(int(n_hidden//2), n_hidden))
        else:
            self.model = nn.Sequential(nn.Linear(input_dim, n_hidden))

        common_modules = nn.Sequential(nn.BatchNorm1d(n_hidden),
                                       nn.ReLU (inplace=True),
                                       nn.Linear(n_hidden, int(np.prod(img_shape))),
                                       nn.Sigmoid())

        self.model.add_module('common_modules', common_modules)

        self.img_shape = img_shape

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *self.img_shape)
        return img

class Classifier(nn.Module):
    def __init__(self, img_shape, invariant_n_hidden, specific_n_hidden, classification_n_hidden, n_classes):
        super(Classifier, self).__init__()

        # specific module
        self.specific = nn.Sequential(nn.Linear(int(np.prod(img_shape)), specific_n_hidden),
                                      nn.ReLU(inplace=True))
        # classification module
        self.classifier_layer = nn.Sequential(nn.Linear(specific_n_hidden + invariant_n_hidden, classification_n_hidden),
                                              nn.ReLU(inplace=True))
        self.output = nn.Linear(classification_n_hidden, n_classes)

    def forward(self, img, invariant):
        discriminative = self.specific(img)
        x = self.classifier_layer(torch.cat([discriminative, invariant],dim=1))
        logits = self.output(x)
        return logits
