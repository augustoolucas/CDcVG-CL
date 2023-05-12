import torch
from torch.nn.functional import one_hot
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.optim import Adam
from models.ff_data import overlay_y_on_x

class CModel(nn.Module):
    def __init__(self, in_channels=1,num_filters=32) -> None:
        super().__init__()
        self.num_filters = num_filters

        self.layers_dict = dict()
        self.layers_dict['linear1'] =  LinearLayer(62, 10, bias=True,name='linear1') #num_filters * 12 * 12
        #self.layers_dict['linear2'] =  LinearLayer(128, 10, bias=True,name='linear2')

        self.max_pool = nn.MaxPool2d(2)
        self.flatten  = nn.Flatten()
        self.relu     = nn.ReLU()

        self.loss_hist = dict()

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            #h = overlay_y_on_x(x, label)
            y_onehot = one_hot(torch.tensor(label).to(torch.int64), 10).cuda()
            y_onehot = y_onehot.to(torch.float)
            h = torch.cat([y_onehot.expand(128, -1), x], dim=1)
            goodness = []
            for layer in self.layers_dict.values():
                h = layer(h)
                goodness = goodness + [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label#.argmax(1)

    def forward(self, x): # X Shape ([bs, 1, 28, 28]) # FOR TESTS
        x = self.relu(x) # X Shape ([bs, 32, 24, 24])
        x = self.max_pool(x) # X Shape ([bs, 32, 12, 12])
        x = self.flatten(x) # X Shape ([bs, 1 * 28 * 28])
        x = self.layers_dict['linear1'](x)  # X Shape ([bs, 128])
        x = self.layers_dict['linear2'](x) # X Shape ([bs, 10])
        return x 
    

    def train(self, x_pos, x_neg,num_epochs=1000):
        h_pos, h_neg = x_pos, x_neg
        for layer_name, layer in self.layers_dict.items():
            print("training layer: ", layer_name)
            h_pos, h_neg, self.loss_hist[layer_name] = layer.train(h_pos, h_neg,num_epochs)
        return self.loss_hist


    def print_net(self):
        print(self)

class LinearLayer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 device='cuda:0', dtype=None,
                   lr: float = 0.03, threshold: float = 2.0,
                     num_epochs: int = 1000, name: str = '') -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=lr)
        self.threshold = threshold
        self.num_epochs = num_epochs
        self.name = name

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_direction = input / (input.norm(2, 1, keepdim=True) + 1e-4)
        #print('trying to mul x with shape = {} and weight with shape = {}'.format(x.shape, self.weight.shape))
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    
    def train(self, x_pos, x_neg,num_epochs):
        losses = []
        for i in tqdm(range(num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)

            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
           
            loss.backward()
            self.opt.step()
            losses.append(loss.detach().cpu())
        
        return self.forward(x_pos).detach(), self.forward(x_neg).detach(), losses


class ConvLayer(nn.Conv2d):
    def __init__():
        pass
    
