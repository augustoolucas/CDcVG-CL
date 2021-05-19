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
    def __init__(self, img_shape, num_latent):
        super(ConvEncoder, self).__init__()
        
        img_channels = img_shape[2]

        self.enc_conv_1 = nn.Conv2d(in_channels=img_channels,
                                    out_channels=16,
                                    kernel_size=(3, 3),
                                    stride=(2, 2))

        self.enc_conv_2 = nn.Conv2d(in_channels=16,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),)

        self.enc_conv_3 = nn.Conv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=(2, 2),
                                    stride=(2, 2))

        self.z_mean = nn.Linear(64*3*3, num_latent)
        self.z_log_var = nn.Linear(64*3*3, num_latent)
        #self.linear = nn.Linear(64*3*3, num_latent)

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

        #x = self.linear(x.view(-1, np.prod(x.size()[1:])))
        #x = F.leaky_relu(x)
        #return x

        z_mean = self.z_mean(x.view(-1, np.prod(x.size()[1:])))
        z_log_var = self.z_log_var(x.view(-1, np.prod(x.size()[1:])))
        encoded = self.reparameterize(z_mean, z_log_var)
        #print('reparameterize out:', encoded.size())

        return encoded, z_mean, z_log_var

    def forward(self, img):
        return self.encoder(img)


class ConvDecoder(nn.Module):
    def __init__(self, img_shape, input_size):
        super(ConvDecoder, self).__init__()

        img_channels = img_shape[2]

        self.dec_linear_1 = nn.Linear(input_size, 64*3*3)

        self.dec_deconv_1 = nn.ConvTranspose2d(in_channels=64,
                                               out_channels=32,
                                               kernel_size=(3, 3),
                                               stride=(2, 2))

        self.dec_deconv_2 = nn.ConvTranspose2d(in_channels=32,
                                               out_channels=16,
                                               kernel_size=(3, 3),
                                               stride=(2, 2),
                                               padding=0)

        self.dec_deconv_3 = nn.ConvTranspose2d(in_channels=16,
                                               out_channels=img_channels,
                                               kernel_size=(4, 4),
                                               stride=(2, 2),
                                               padding=0)

    def decoder(self, encoded):
        x = self.dec_linear_1(encoded)
        x = x.view(-1, 64, 3, 3)
        #print('dec_linear_1 out:', x.size())

        x = self.dec_deconv_1(x)
        x = F.leaky_relu(x)
        #print('deconv1 out:', x.size())

        x = self.dec_deconv_2(x)
        x = F.leaky_relu(x)
        #print('deconv2 out:', x.size())

        x = self.dec_deconv_3(x)
        x = F.leaky_relu(x)
        #print('deconv3 out:', x.size())

        decoded = torch.sigmoid(x)
        return decoded

    def forward(self, encoded):
        return self.decoder(encoded)


class Encoder(nn.Module):
    def __init__(self, img_shape, n_hidden, latent_dim, n_layers=1):
        super(Encoder, self).__init__()

        self.model = nn.Sequential()

        input_dim = int(np.prod(img_shape))

        for hl in range(n_layers):
            output_dim = int(n_hidden//2**hl)
            new_module = nn.Sequential(nn.Linear(input_dim, output_dim),
                                       nn.BatchNorm1d(output_dim),
                                       nn.ReLU(inplace=True))

            self.model.add_module(f'layer_{hl}', new_module)

            input_dim = output_dim

        self.mu = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)
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
                 layer_size,
                 latent_dim,
                 n_classes,
                 use_label=True,
                 n_layers=1):
        super(Decoder, self).__init__()
        # conditional generation
        if use_label:
            input_dim = latent_dim + n_classes
        else:
            input_dim = latent_dim

        self.model = nn.Sequential()

        for hl in range(1, n_layers+1):
            output_dim = int(layer_size//2**(n_layers-hl))
            new_module = nn.Sequential(nn.Linear(input_dim, output_dim),
                                       nn.BatchNorm1d(output_dim),
                                       nn.ReLU(inplace=True))
            self.model.add_module(f'layer_{hl}', new_module)

            input_dim = output_dim

        output_layer = nn.Sequential(nn.Linear(input_dim, int(np.prod(img_shape))),
                                     nn.Sigmoid())

        self.model.add_module('output_layer', output_layer)

        self.img_shape = img_shape

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *self.img_shape)
        return img


class ResidualBlock(nn.Module):
    """A simple residual block."""

    def __init__(self, n_channels, hidden_channels):
        """Initializes a new ResidualBlock instance.
        Args:
            n_channels: Number of input and output channels.
            hidden_channels: Number of hidden channels.
        """
        super().__init__()
        self._net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=n_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        return x + self._net(x)


class ResidualStack(nn.Module):
    """A stack of multiple ResidualBlocks."""

    def __init__(self, n_channels, hidden_channels, n_residual_blocks=1):
        """Initializes a new ResidualStack instance.
        Args:
            n_channels: Number of input and output channels.
            hidden_channels: Number of hidden channels.
            n_residual_blocks: Number of residual blocks in the stack.
        """
        super().__init__()
        self._net = nn.Sequential(
            *[
                ResidualBlock(n_channels, hidden_channels)
                for _ in range(n_residual_blocks)
            ]
            + [nn.ReLU()]
        )

    def forward(self, x):
        return self._net(x)


class VQVAE_Encoder(nn.Module):
    """A feedforward encoder which downsamples its input."""

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        n_residual_blocks,
        residual_channels,
        stride,
        num_latent,
    ):
        """Initializes a new Encoder instance.
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            hidden_channels: Number of channels in non residual block hidden layers.
            n_residual_blocks: Number of residual blocks in each residual stack.
            residual_channels: Number of hidden channels in residual blocks.
            stride: Stride to use in the downsampling convolutions. Must be even.
        """
        super().__init__()
        assert stride % 2 == 0, '"stride" must be even.'

        net = []
        for i in range(stride // 2):
            first, last = 0, stride // 2 - 1
            in_c = in_channels if i == first else hidden_channels // 2
            out_c = hidden_channels // 2 if i < last else hidden_channels
            net.append(
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            net.append(nn.ReLU())

        if n_residual_blocks > 0:
            net.append(
                ResidualStack(
                    n_channels=hidden_channels,
                    hidden_channels=residual_channels,
                    n_residual_blocks=n_residual_blocks,
                )
            )

        net.append(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )
        )
        self._net = nn.Sequential(*net)

        self.z_mean = nn.Linear(10*8*8, num_latent)
        self.z_log_var = nn.Linear(10*8*8, num_latent)

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(device)
        z = z_mu + eps * torch.exp(z_log_var/2.).to(device)
        return z

    def forward(self, x):
        x = self._net(x)

        z_mean = self.z_mean(x.view(-1, np.prod(x.size()[1:])))
        z_log_var = self.z_log_var(x.view(-1, np.prod(x.size()[1:])))
        encoded = self.reparameterize(z_mean, z_log_var)

        return encoded, z_mean, z_log_var


class VQVAE_Decoder(nn.Module):
    """A feedforward encoder which upsamples its input."""

    def __init__(
        self,
        input_size,
        in_channels,
        out_channels,
        hidden_channels,
        n_residual_blocks,
        residual_channels,
        stride,
    ):
        """Initializes a new Decoder instance.
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            hidden_channels: Number of channels in (non residual block) hidden layers.
            n_residual_blocks: Number of residual blocks in each residual stack.
            residual_channels: Number of hidden channels in residual blocks.
            stride: Stride to use in the upsampling (i.e. transpose) convolutions. Must
                be even.
        """
        super().__init__()

        assert stride % 2 == 0, '"stride" must be even.'

        net = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            ResidualStack(
                n_channels=hidden_channels,
                hidden_channels=residual_channels,
                n_residual_blocks=n_residual_blocks,
            ),
        ]
        for i in range(stride // 2):
            first, last = 0, stride // 2 - 1
            in_c = hidden_channels if i == first else hidden_channels // 2
            out_c = hidden_channels // 2 if i < last else out_channels
            net.append(
                nn.ConvTranspose2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            if i < last:
                net.append(nn.ReLU())
        self._net = nn.Sequential(*net)

        self.linear = nn.Linear(input_size, 10*8*8)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 10, 8, 8)
        return self._net(x)


class ConvClassifier(nn.Module):
    def __init__(self,
                 img_shape,
                 specific_size,
                 n_layers_classifier,
                 classification_layer_size,
                 n_classes,
                 latent_dim):
        super(ConvClassifier, self).__init__()

        # Specific Module
        img_channels = img_shape[2]

        self.specific_conv = nn.Sequential(nn.Conv2d(in_channels=img_channels,
                                                     out_channels=16,
                                                     kernel_size=(6, 6),
                                                     stride=(2, 2)),
                                           nn.LeakyReLU(inplace=True),
                                           nn.Conv2d(in_channels=16,
                                                     out_channels=32,
                                                     kernel_size=(4, 4),
                                                     stride=(2, 2)),
                                           nn.LeakyReLU(inplace=True),
                                           nn.Conv2d(in_channels=32,
                                                     out_channels=64,
                                                     kernel_size=(2, 2),
                                                     stride=(2, 2)))

        self.specific_linear = nn.Sequential(nn.Linear(64*2*2,
                                                       specific_size),
                                             nn.LeakyReLU(inplace=True))

        # Classifier Module
        input_dim = specific_size + latent_dim
        self.classifier = nn.Sequential()

        n_layers_classifier -= 1
        for hl in range(1, n_layers_classifier+1):
            output_dim = int(classification_layer_size//2**(n_layers_classifier-hl))
            new_module = nn.Sequential(nn.Linear(in_features=input_dim,
                                                 out_features=output_dim),
                                       nn.ReLU(inplace=True))

            self.classifier.add_module(f'layer_{hl}', new_module)
            input_dim = output_dim

        output_layer = nn.Sequential(nn.Linear(in_features=input_dim,
                                               out_features=n_classes),
                                     nn.Softmax())

        self.classifier.add_module('output_layer', output_layer)

    def forward(self, img, invariant):
        x = self.specific_conv(img)
        x = self.specific_linear(x.view(x.size(0), -1))
        x = self.classifier(torch.cat([x, invariant], dim=1))

        return x


class Classifier(nn.Module):
    def __init__(self,
                 img_shape,
                 invariant_size,
                 specific_size,
                 classification_layer_size,
                 n_classes,
                 specific_layers=0,
                 n_layers_classifier=1):
        super(Classifier, self).__init__()

        # specific module
        self.specific = nn.Sequential()

        input_dim = int(np.prod(img_shape))

        for hl in range(1, specific_layers+1):
            output_dim = int(specific_size//2**(specific_layers-hl))
            new_module = nn.Sequential(nn.Linear(input_dim, output_dim),
                                       nn.ReLU(inplace=True))

            self.specific.add_module(f'layer_{hl}', new_module)

            input_dim = output_dim

        # classification module
        self.classifier = nn.Sequential()
        input_dim = input_dim + invariant_size
        n_layers_classifier -= 1
        for hl in range(1, n_layers_classifier+1):
            output_dim = int(classification_layer_size//2**(n_layers_classifier-hl))
            new_module = nn.Sequential(nn.Linear(in_features=input_dim,
                                                 out_features=output_dim),
                                       nn.ReLU(inplace=True))

            self.classifier.add_module(f'layer_{hl}', new_module)
            input_dim = output_dim

        output_layer = nn.Sequential(nn.Linear(in_features=input_dim,
                                               out_features=n_classes),
                                     nn.Softmax())

        self.classifier.add_module('output_layer', output_layer)

    def forward(self, img, invariant):
        discriminative = self.specific(img.view(img.size(0), -1))
        x = self.classifier(torch.cat([discriminative, invariant],dim=1))
        return x
