from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch
import copy
import random

def onehot_encoder(labels, n_classes):
    labels_onehot = torch.zeros(labels.shape[0], n_classes)
    labels_onehot.scatter_(1, labels.view(-1, 1), 1)

    return labels_onehot

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
    task_labels = [[x, x+1] for x in range(0, n_classes, 2)]
    datasets = []

    for labels in task_labels:
        idxs = np.in1d(dataset.targets, labels)
        task_set = copy.deepcopy(dataset)
        task_set.targets = dataset.targets[idxs]
        task_set.data = dataset.data[idxs]
        datasets.append(task_set)
     
    return datasets


def get_tasks_labels(tasks):
    tasks_labels = [list(set(task.targets.tolist())) for task in tasks]
    return tasks_labels


def get_task_data_shape(tasks):
    return tasks[0][0][0].shape


def get_tasks_classes(tasks):
    classes = [class_ for task in tasks for class_ in set(task.targets.tolist())]
    return classes


def get_dataloader(dataset, batch_size):
    if type(dataset) == list:
        data = torch.tensor([], dtype=dataset[0].data.dtype)
        targets = torch.tensor([])

        for ds in dataset:
            data = torch.cat([data, ds.data])
            targets = torch.cat([targets, ds.targets])

        dataset = copy.deepcopy(dataset[0])
        dataset.data = data
        dataset.targets = targets

    loader = DataLoader(dataset,
                        batch_size,
                        num_workers=0,
                        pin_memory=True,
                        shuffle=True)
    return loader

def update_train_set(dataset, new_images, new_labels):
    if new_images.size(0) == 0:
        return dataset

    if len(dataset.data.shape) < len(new_images.shape):
        if new_images.size(1) == 1:
            new_images = torch.squeeze(new_images, dim=1)
        else:
            # TODO: Raise an exception instead
            print('Unknown error')
            exit()

    if dataset.data.dtype != new_images.dtype:
        if dataset.data.dtype == torch.uint8 and new_images.dtype == torch.float32:
            new_images = (new_images * 255).type(dataset.data.dtype)
        else:
            print('Warning: Mismatched data types.')

    dataset.data = torch.cat([dataset.data, new_images], dim=0)
    dataset.targets = torch.cat([dataset.targets, new_labels], dim=0)

    return dataset

