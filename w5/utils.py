# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/utils.py
#   + License: MIT
from __future__ import division
import math
import random
import scipy.misc
import numpy as np
import skimage.transform
import skimage.io
from matplotlib import pyplot as plt

def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)

def save_images(images, size, image_path, show=False):
    return imsave(inverse_transform(images), size, image_path, show)

def imread(path):
    return skimage.io.imread(path)

def imsave(images, size, path, show):
    figure = plt.figure(figsize=(20, 20))
    for i, image in enumerate(images):
        plt.subplot(5, 5, i+1)
        plt.imshow(image) # (255*img).astype(np.uint8)
        plt.axis('off')
        
    if not show:
        with open(path, 'wb') as file:
            figure.savefig(file, bbox_inches='tight')
            plt.close(figure)
    else:
        plt.show()
    
    #for i, img in enumerate(images):
    #    skimage.io.imsave(paths[i], (255*img).astype(np.uint8) )

def transform(image, final_size, is_crop=True):
    if is_crop:
        size = image.shape[1]
        if image.shape[0] > size:
            crop_size = image.shape[0] - size
            top_crop = int(round( crop_size * .65 ))
            bottom_crop = crop_size - top_crop
            image = image[ top_crop : -bottom_crop, :]
    image = skimage.transform.resize( image, (final_size, final_size), anti_aliasing=True)
    image = image * 2. - 1.
    return image

def inverse_transform(images):
    return (np.array(images) + 1.) / 2.