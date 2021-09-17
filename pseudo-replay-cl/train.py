import torch
import data_utils

def train():
    ### Load Data ###
    train_tasks, val_tasks, test_tasks = data_utils.load_tasks('MNIST', val=True)

if __name__ == '__main__':
    train()

