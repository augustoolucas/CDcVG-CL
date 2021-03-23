# Author: Ghada Sokar et al.
# This is the implementation for the Learning Invariant Representation for Continual Learning paper in AAAI workshop on Meta-Learning for Computer Vision

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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

def load_data(dataset):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == "MNIST":
        full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    elif dataset == "EMNIST":
        full_dataset = datasets.EMNIST('./data', split="mnist", train=True, download=True, transform=transform)
        test_dataset = datasets.EMNIST('./data', split="mnist", train=False, transform=transform)
    elif dataset == "Fashion-MNIST":
        full_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)
    elif dataset == "CIFAR10":
        full_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
        #full_dataset.data = full_dataset.data.astype(np.uint8)
        #test_dataset.data = test_dataset.data.astype(np.uint8)
        full_dataset.targets = torch.Tensor(full_dataset.targets).type(torch.int64)
        test_dataset.targets = torch.Tensor(test_dataset.targets).type(torch.int64)
    elif dataset == "SVHN":
        full_dataset = datasets.SVHN('./data', split='train', download=True, transform=transform)
        test_dataset = datasets.SVHN('./data', split='test', download=True, transform=transform)
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

