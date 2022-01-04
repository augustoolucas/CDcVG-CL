import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from imageio import imwrite
from matplotlib.ticker import MaxNLocator
from torchvision.transforms import ToPILImage

def show_image(image):
    ToPILImage()(image).show()


def tsne_plot(tsne_results, labels, path, title=''):
    plt.figure(figsize=(16,10))
    sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], hue=labels, legend='full', palette=sns.color_palette('hls', len(set(labels))))
    plt.title(title)
    plt.savefig(dpi=150,
                fname=path,
                bbox_inches='tight')
    plt.close()


def plot_losses(losses, xlabel, ylabel, title, fname, fontsize=4.5):
    fig = plt.figure()
    plt.scatter(list(range(len(losses))), losses)
    plt.plot(list(range(len(losses))), losses, linestyle='dashed', alpha=0.25)
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for i,j in enumerate(losses):
        ax.annotate(f'{j:.02f}',
                    xy=(i-len(losses)*0.01, j+(max(losses) - min(losses))*0.02),
                    fontsize=fontsize)
    plt.ylim(top=max(losses) + (max(losses) - min(losses))*0.1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(dpi=300,
                fname=fname,
                bbox_inches='tight')
    plt.close()

def multi_plots(data1, data2, xlabel, ylabel1, ylabel2, title, fname):
    fig, ax1 = plt.subplots()
    color = 'k'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=color)
    ax1.scatter(list(range(len(data1))), data1, color=color)
    ax1.plot(list(range(len(data1))), data1, linestyle='dashed', alpha=0.25, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(ylabel2, color=color)  # we already handled the x-label with ax1
    ax2.scatter(list(range(len(data2))), data2, color=color)
    ax2.plot(list(range(len(data2))), data2, linestyle='dashed', alpha=0.25, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(title)
    plt.savefig(dpi=300,
                fname=fname,
                bbox_inches='tight')
    plt.close()


def visualize(real_images, recon_images, gen_images, task_id, path):
    shape = gen_images[0].shape
    shape = shape if shape[0] > shape[2] else (shape[1], shape[2], shape[0])
    cmap = 'gray' if shape[2] == 1 else None
    plotter = plot_samples('./', shape, 8, 8)
    real_images = plotter.get_images(real_images[:64])
    recon_images = plotter.get_images(recon_images[:64])
    gen_images = plotter.get_images(gen_images[:64])
    original = real_images.astype(np.uint8)
    reconstructed = recon_images.astype(np.uint8)
    samples = gen_images.astype(np.uint8)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(original, cmap=cmap)
    ax1.set_xlabel('Original')
    ax2.imshow(reconstructed, cmap=cmap)
    ax2.set_xlabel('Reconstructed')
    ax3.imshow(samples, cmap=cmap)
    ax3.set_xlabel('Generated')

    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.spines.left.set_visible(False)

    plt.savefig(dpi=500,
                fname=f'{path}/images.png',
                bbox_inches='tight')
    plt.close()


class plot_samples():
    def __init__(self, DIR, img_shape, n_img_x=8, n_img_y=8):
        img_w, img_h, n_channels = img_shape if img_shape[0] > img_shape[2] else (img_shape[1], img_shape[2], img_shape[0])
        self.DIR = DIR
        assert n_img_x > 0 and n_img_y > 0
        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_channels = n_channels
        self.n_total_imgs = n_img_x * n_img_y
        assert img_w > 0 and img_h > 0
        self.img_w = img_w
        self.img_h = img_h

    def get_images(self, images):
        images = images.reshape(self.n_total_imgs, self.n_channels, self.img_h, self.img_w)
        return self._merge(images, [self.n_img_y, self.n_img_x, self.n_channels])

    def save_images(self, images, name='result.jpg'):
        images = images.reshape(self.n_total_imgs, self.n_channels, self.img_h, self.img_w)
        imwrite(self.DIR + "/"+name, self._merge(images, [self.n_img_y, self.n_img_x, self.n_channels]))

    def _merge(self, images, size):
        if self.n_channels == 3:
            c, h, w = images.shape[1], images.shape[2], images.shape[3]
            img = np.zeros((h * size[0], w * size[1], c))
        else:
            h, w = images.shape[2], images.shape[3]
            img = np.zeros((h * size[0], w * size[1]))

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])

            image_ = ToPILImage(mode='RGB')(image) if self.n_channels == 3 else ToPILImage()(image)
            image_ = np.array(image_.resize((w, h), Image.BICUBIC))

            img[j*h:j*h+h, i*w:i*w+w] = image_
        
        return img
