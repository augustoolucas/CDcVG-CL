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

def load_data(dataset, train=True):
    if dataset == 'MNIST':
        data = datasets.MNIST(root='./Datasets',
                              download=True,
                              train=train,
                              transform=transforms.ToTensor())
    elif dataset == 'CIFAR10':
        data = datasets.CIFAR10(root='./Datasets',
                                download=True,
                                train=train,
                                transform=transforms.ToTensor())

    elif dataset == 'FashionMNIST':
        data = datasets.FashionMNIST(root='./Datasets',
                                     download=True,
                                     train=train,
                                     transform=transforms.ToTensor())
    else:
        print('Invalid dataset.')
        exit()

    return data


def load_tasks(dataset, balanced=False, train=True):
    data = load_data(dataset, train)
    tasks = create_tasks(data, balanced)

    return tasks


def create_tasks(dataset, balanced=False):
    if type(dataset.targets) is list:
        n_classes = len(set(dataset.targets))
    else:
        n_classes = len(set(dataset.targets.tolist()))

    task_labels = [[x, x+1] for x in range(0, n_classes, 2)]
    datasets = []

    for labels in task_labels:
        idxs = []
        for label in labels:
            all_idxs = np.nonzero(np.isin(dataset.targets, [label]))[0]
            all_idxs = all_idxs[:5000] if balanced else all_idxs
            idxs.extend(all_idxs)
        task_set = copy.deepcopy(dataset)
        if type(dataset.targets) is list:
            task_set.targets = np.array(dataset.targets)[idxs].tolist()
        else:
            task_set.targets = dataset.targets[idxs]
        task_set.data = dataset.data[idxs]
        datasets.append(task_set)
     
    return datasets


def get_labels(tasks):
    tasks_labels = [list(set(np.array(task.targets).tolist())) for task in tasks]
    return tasks_labels


def get_img_shape(tasks):
    return tasks[0][0][0].shape


def get_classes(tasks):
    classes = [class_ for task in tasks for class_ in set(np.array(task.targets).tolist())]
    return classes


def get_dataloader(dataset, batch_size):
    data_ndarray = False
    targets_list = False
    if type(dataset) == list:
        if type(dataset[0].data) is np.ndarray:
            data_ndarray = True
            data = None
        else:
            data = torch.tensor([], dtype=dataset[0].data.dtype)

        if type(dataset[0].targets) is list:
            targets_list = True
            targets= []
        else:
            targets = torch.tensor([])

        for ds in dataset:
            if data_ndarray:
                data = np.vstack([data, ds.data]) if data is not None else ds.data
            else:
                data = torch.cat([data, ds.data])

            if targets_list:
                targets = targets + ds.targets
            else:
                targets = torch.cat([targets, ds.targets])

        dataset = copy.deepcopy(dataset[0])
        dataset.data = data
        dataset.targets = targets

    loader = DataLoader(dataset,
                        batch_size,
                        num_workers=4,
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

    if dataset.data[0].shape != new_images[0].shape:
        new_images = torch.swapaxes(new_images, 1, 2)
        new_images = torch.swapaxes(new_images, 2, 3)

    if type(dataset.data) is np.ndarray:
        new_images = (new_images * 255).numpy().astype(dataset.data.dtype)
    elif dataset.data.dtype != new_images.dtype:
        if dataset.data.dtype == torch.uint8 and new_images.dtype == torch.float32:
            new_images = (new_images * 255).type(dataset.data.dtype)
        else:
            print('Warning: Mismatched data types.')
            exit()

    if type(dataset.data) is np.ndarray:
        dataset.data = np.vstack([dataset.data, new_images])
        dataset.targets += new_labels.tolist()
    else:
        dataset.data = torch.cat([dataset.data, new_images], dim=0)
        dataset.targets = torch.cat([dataset.targets, new_labels], dim=0)

    return dataset

