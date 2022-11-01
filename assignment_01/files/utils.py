import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def load_image(path_to_image):
    return np.asarray(Image.open(path_to_image))[:,:,:3]/255.

def show_image(image, title=None):
    plt.figure()
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.imshow(image, cmap='gray',vmin=0., vmax=1.)
    plt.show(block=False)

def show_multiple_images(images):
    n = len(images)
    fig, axs = plt.subplots(1, n, figsize=(12,3))
    for ax, image in zip(axs.reshape((-1,)), images):
        ax.axis('off')
        ax.imshow(image, cmap='gray',vmin=0., vmax=1.)
    plt.show(block=False)
    
def plot_hist(hist):
    plt.figure(figsize=(16,9))
    plt.title('Image Histogram')
    plt.xlabel('luminance')
    plt.ylabel('pixel count')
    plt.xlim([0.0, 256.0])
    plt.bar(np.arange(256), hist)
    plt.show(block=False)
    
def show_image_with_hist(image_gray, hist):
    fig, (ax_hist, ax_image) = plt.subplots(1, 2, figsize=(16,5))
    ax_image.axis('off')
    plt.imshow(image_gray, cmap='gray',vmin=0., vmax=1.)
    
    ax_hist.set_title('Image Histogram')
    ax_hist.set_xlabel('luminance')
    ax_hist.set_ylabel('pixel count')
    ax_hist.set_xlim([0.0, 256.0])
    ax_hist.bar(np.arange(256), hist)
    plt.show(block=False)
    
def check_arrays(title, keys, arrays, arrays_true):
    print(f'Checking {title}:')
    for key, array, array_true in zip(keys, arrays, arrays_true):
        result = 'passed' if np.isclose(array, array_true).all() else 'failed'
        print(f'{key}: {result}')
    print()
