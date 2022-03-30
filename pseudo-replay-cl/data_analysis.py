import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import data

def main():
    ds = 'FashionMNIST'
    dataset = data.load_data(ds, train=True)
    plt.Figure()
    hist = plt.hist(torch.flatten(dataset.data, start_dim=0).numpy())
    plt.title(ds)
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')
    plt.savefig(f'./{ds}_hist.jpg')

if __name__ == '__main__':
    main()
