# Author: Ghada Sokar et al.
# This is the implementation for the Learning Invariant Representation for Continual Learning paper in AAAI workshop on Meta-Learning for Computer Vision

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import svhn
from tqdm import tqdm
import copy

# Fix for Downloading MNIST giving HTTP Error 403:
# https://github.com/pytorch/vision/issues/1938#issuecomment-594623431
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

def get_train_loader(train_dataset,batch_size):
    train_loader = DataLoader(
    train_dataset,
    batch_size,
    num_workers=0,
    pin_memory=True, shuffle=True)
    return train_loader

def get_test_loader(test_dataset,test_batch_size):
    test_loader = DataLoader(
        test_dataset,
        test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    return test_loader

class CIFAR10(datasets.CIFAR10):
    def __init__(self, path, train=True, download=True, grayscale=False, res=32):
        super().__init__(path, train=train, download=download)
        aux = []
        for data in self.data:
            img = Image.fromarray(data)

            if grayscale:
                img = img.convert('L')           
            
            if img.size != (res, res):
                img = img.resize((res, res))

            array = np.asarray(img)
            aux.append(array)

        self.grayscale = grayscale
        self.res = res
        self.data = np.asarray(aux)
        self.targets = torch.Tensor(self.targets).type(torch.int64)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]

        if self.grayscale:
            img = Image.fromarray(img, mode='L')
        else:
            img = Image.fromarray(img)
        
        img = transforms.ToTensor()(img)    
        return img, target

class MNIST_RGB(datasets.MNIST):
    def __init__(self, path, train=True, download=True, transform=None, background_data=None):
        super().__init__(path, train=train, download=download, transform=transform)
        breakpoint()
        background_data = torch.Tensor(background_data).permute(0,3,1,2)
        self.data = self.data.unsqueeze(1).repeat(1, 3, 1, 1)
        tmp = background_data[0] * self.data[0]
        tmp = tmp/2
        breakpoint()

def load_data(dataset):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == "MNIST":
        full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        breakpoint()

    if dataset == "MNIST-RGB":
        cifar = CIFAR10('./data', train=True, download=True, grayscale=False, res=28)
        full_dataset = MNIST_RGB('./data', train=True, download=True, transform=transform, background_data=cifar.data)
        test_dataset = MNIST_RGB('./data', train=False, transform=transform, background_data=cifar.data)

        
    elif dataset == "EMNIST":
        full_dataset = datasets.EMNIST('./data', split="letters", train=True, download=True, transform=transform)
        test_dataset = datasets.EMNIST('./data', split="letters", train=False, transform=transform)
        full_dataset.data = torch.transpose(full_dataset.data, 1, 2)
        test_dataset.data = torch.transpose(test_dataset.data, 1, 2)

    elif dataset == "Fashion-MNIST":
        full_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)

    elif dataset == "CIFAR10":
        full_dataset = CIFAR10('./data', train=True, download=True, grayscale=True, res=28)
        test_dataset = CIFAR10('./data', train=False, grayscale=True, res=28)
       
    elif dataset == "SVHN":
        full_dataset = svhn.SVHN('./data', split='train', download=True, transform=transform)
        test_dataset = svhn.SVHN('./data', split='test', download=True, transform=transform)
        #full_dataset.data = full_dataset.data.swapaxes(1, 3)
        #test_dataset.data = test_dataset.data.swapaxes(1, 3)
        full_dataset.targets = full_dataset.labels
        test_dataset.targets = test_dataset.labels

    else:
        print("Invalid Dataset")
        exit()

    return full_dataset,test_dataset

def task_construction(task_labels, dataset):
    full_dataset,test_dataset = load_data(dataset)
    train_dataset = split_dataset_by_labels(full_dataset, task_labels)
    test_dataset = split_dataset_by_labels(test_dataset, task_labels)
    return train_dataset,test_dataset

def split_dataset_by_labels(dataset, task_labels):
    datasets = []
    task_idx = 0
    for labels in task_labels:
        idx = np.in1d(dataset.targets, labels)
        splited_dataset = copy.deepcopy(dataset)
        splited_dataset.targets = splited_dataset.targets[idx]
        splited_dataset.data = splited_dataset.data[idx]
        datasets.append(splited_dataset)
        task_idx += 1
    return datasets

