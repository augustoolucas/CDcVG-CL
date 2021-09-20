from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch
import copy
import random

def load_data(dataset, val=False):
    if dataset == 'MNIST':
        train_set = datasets.MNIST(root='./Datasets',
                                   download=True,
                                   train=True,
                                   transform=transforms.ToTensor())

        test_set = datasets.MNIST(root='./Datasets',
                                  download=True,
                                  train=False,
                                  transform=transforms.ToTensor())

    else:
        print('Invalid dataset.')
        exit()

    val_set = None
    if val:
        val_set_size = int(.10*len(train_set))
        idxs = list(range(len(train_set)))
        random.shuffle(idxs)
        val_idxs, test_idxs = idxs[:val_set_size], idxs[val_set_size:]

        val_set = copy.deepcopy(train_set)
        val_set.data = val_set.data[val_idxs]
        val_set.targets = val_set.targets[val_idxs]

        train_set.data = train_set.data[test_idxs]
        train_set.targets = train_set.targets[test_idxs]

    return train_set, val_set, test_set

def load_tasks(dataset, val=False):
    train_set, val_set, test_set = load_data(dataset, val)
    train_task = create_tasks(train_set)
    val_task = create_tasks(val_set) if val else None
    test_task = create_tasks(test_set)

    return train_task, val_task, test_task

def create_tasks(dataset):
    n_classes = len(set(dataset.targets.tolist()))
    task_labels = get_tasks_labels(n_classes)
    datasets = []

    for labels in task_labels:
        idxs = np.in1d(dataset.targets, labels)
        task_set = copy.deepcopy(dataset)
        task_set.targets = dataset.targets[idxs]
        task_set.data = dataset.data[idxs]
        datasets.append(task_set)
     
    return datasets

def get_tasks_labels(n_classes):
    return [[x, x+1] for x in range(0, n_classes, 2)]

def get_task_data_shape(data):
    return data[0].data.shape[1:]

def get_task_n_classes(data):
    n_classes = 0
    for task in data:
        n_classes += len(set(task.targets.tolist()))
        
    return n_classes

def get_dataloader(dataset, batch_size):
    loader = DataLoader(dataset,
                        batch_size,
                        num_workers=0,
                        pin_memory=True,
                        shuffle=True)
    return loader

def update_train_set(dataset, new_images, new_labels):
    if new_images.size(0) == 0:
        return dataset

    dataset.data = torch.cat([dataset.data, new_images], dim=0)
    dataset.targets = torch.cat([dataset.targets, new_labels], dim=0)

    return dataset

