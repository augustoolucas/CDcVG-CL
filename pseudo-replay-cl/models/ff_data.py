import torch 
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
import matplotlib.pyplot as plt



def overlay_y_on_x(x, y, classes=10):
    x_ = x.clone()
    x_[:, :classes] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_

def MNIST_loaders(train_batch_size=500, test_batch_size=100):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

def format_data(x, y):
    #x, y = next(iter(loader))
    x, y = x.cuda(), y.cuda()
    #x_pos = overlay_y_on_x(x, y)
    y_onehot = one_hot(y, 10).cuda()
    x_pos = torch.cat([y_onehot, x], dim=1)
    rnd = torch.randperm(x.size(0))
    #x_neg = overlay_y_on_x(x, y[rnd])
    x_neg = torch.cat([y_onehot[rnd], x], dim=1)
    return x_pos, x_neg
