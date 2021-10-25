import numpy as np
import torch
from torchvision.transforms import ToPILImage
from imageio import imwrite
import matplotlib.pyplot as plt
from PIL import Image

def show_image(image):
    ToPILImage()(image).show()

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
                fname='%s/imgs_task_%d.jpg' % (path, task_id),
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
