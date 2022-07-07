from torch.utils.data import DataLoader, ConcatDataset
from avalanche.benchmarks import SplitMNIST, SplitCIFAR10, SplitFMNIST
from avalanche.benchmarks.generators import benchmark_with_validation_stream
from torchvision import datasets, transforms
import numpy as np
import torch
import copy
import random

def onehot_encoder(labels, n_classes):
    labels_onehot = torch.zeros(labels.shape[0], n_classes)
    labels_onehot.scatter_(1, labels.view(-1, 1), 1)

    return labels_onehot

def load_data(cfg, n_tasks):
    if cfg['dataset'] == 'MNIST':
        data = SplitMNIST(n_experiences=n_tasks,
                          dataset_root='./Datasets',
                          return_task_id=False,
                          shuffle=False,
                          train_transform=transforms.ToTensor(),
                          eval_transform=transforms.ToTensor())
    elif cfg['dataset'] == 'CIFAR10':
        data = SplitCIFAR10(n_experiences=n_tasks,
                            dataset_root='./Datasets',
                            return_task_id=False,
                            shuffle=False,
                            train_transform=transforms.ToTensor(),
                            eval_transform=transforms.ToTensor())
    elif cfg['dataset'] == 'CIFAR10-Gray':
        transfs = transforms.Compose([transforms.ToTensor(),
                                      transforms.Grayscale(1)])
        data = SplitCIFAR10(n_experiences=n_tasks,
                            dataset_root='./Datasets',
                            return_task_id=False,
                            shuffle=False,
                            train_transform=transfs,
                            eval_transform=transfs)
    elif cfg['dataset'] == 'FashionMNIST':
        data = SplitFMNIST(n_experiences=n_tasks,
                           dataset_root='./Datasets',
                           return_task_id=False,
                           shuffle=False,
                           train_transform=transforms.ToTensor(),
                           eval_transform=transforms.ToTensor())
    else:
        raise ValueError(f'Invalid dataset {cfg["dataset"]}')

    if cfg['use_validation_set']:
        n_classes = data.n_classes
        classes_order = data.classes_order
        data = benchmark_with_validation_stream(benchmark_instance=data,
                                                validation_size=cfg['val_size'])
        data.n_classes = n_classes
        data.classes_order = classes_order

    return data


def get_dataloader(dataset, batch_size):
    loader = DataLoader(dataset,
                        batch_size,
                        num_workers=4,
                        pin_memory=True,
                        shuffle=True)
    return loader
