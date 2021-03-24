import numpy as np
import torch
from torchvision import transforms
from imageio import imwrite
from PIL import Image

imsave = imwrite
# this function is borrowed from https://github.com/hwalsuklee/tensorflow-mnist-AAE/blob/master/plot_utils.py
class plot_samples():
    def __init__(self, DIR, n_img_x=8, n_img_y=8, img_w=28, img_h=28, n_channels=1):
        self.DIR = DIR
        assert n_img_x > 0 and n_img_y > 0
        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_channels = n_channels
        self.n_total_imgs = n_img_x * n_img_y
        assert img_w > 0 and img_h > 0
        self.img_w = img_w
        self.img_h = img_h

    def save_images(self, images, name='result.jpg'):
        images = images.reshape(self.n_total_imgs, self.n_channels, self.img_h, self.img_w)
        imsave(self.DIR + "/"+name, self._merge(images, [self.n_img_y, self.n_img_x, self.n_channels]))

    def _merge(self, images, size):
        if isRGB(images[0]):
            c, h, w = images.shape[1], images.shape[2], images.shape[3]
            img = np.zeros((h * size[0], w * size[1], c))
        else:
            h, w = images.shape[2], images.shape[3]
            img = np.zeros((h * size[0], w * size[1]))

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])
            
            image_ = transforms.ToPILImage(mode='RGB')(image) if isRGB(image) else transforms.ToPILImage()(image)
            image_ = np.array(image_.resize((w, h), Image.BICUBIC))

            img[j*h:j*h+h, i*w:i*w+w] = image_
        
        return img

def isRGB(image):
    return image.shape[0] == 3