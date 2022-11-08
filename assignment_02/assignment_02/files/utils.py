import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def gauss_function(x, mu, sigma):
    '''Implements a multi-dimensional bell-curve-function proportional to the gaussian probability density.
    
    See also https://en.wikipedia.org/wiki/Gaussian_function.
    
    Args:
        x: An arbitrary numpy array with shape(n, d) containing n vectors of dimension d.
           The function is evaluated at each vector.
        mu: A numpy-array containing the translation parameter.
        sigma: A positive float defining the scale. Larger values cause wider curves.

    Returns:
        A numpy array with shape (d,) containing the function values.
    '''
    return np.exp(-((x - mu)**2).sum(axis=-1) / (2 * sigma**2))

def load_images(paths_to_images):
    return [load_image(path_to_image) for path_to_image in paths_to_images]

def load_image(path_to_image):
    with Image.open(path_to_image) as image_file:
        image_array = np.asarray(image_file)[:,:,:3]/255.
    return image_array

def show_image(image, title=None, block=False):
    plt.figure()
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.imshow(image, cmap='gray',vmin=0., vmax=1.)
    plt.show(block=block)

def show_multiple_images(images, titles=None):
    n = len(images)
    if titles is None:
        titles = n * [None]
    fig, axs = plt.subplots(1, n, figsize=(n*(5.5),5))
    for ax, title, image in zip(axs.reshape((-1,)), titles, images):
        if title is not None:
            ax.set_title(title)
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
