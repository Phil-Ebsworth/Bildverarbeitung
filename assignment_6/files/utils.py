import numpy as np
from skimage import feature

from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

def gauss_filter(image, w, sigma, stride_x=1, stride_y=1):
    """Applies gauss filtering to the input image.

    Args:
        image: A numpy array with shape (height, width, channels) representing the imput image.
        w: Defines the patch size 2*w+1 of the filter.
        stride_x: The step-size in x-direction.
        stride_y: The step-size in y-direction.

    Returns:
        A numpy array with shape (⌈height/stride⌉, ⌈width/stride⌉, channels) representing the filtered image.
        Note that the input image is zero-padded to preserve the original resolution.
    """

    gauss_kern_1d = gauss_function(np.arange(-w,w+1).reshape(-1,1),0,sigma)
    gauss_kern_1d /= gauss_kern_1d.sum()
    return conv1d(
        image=conv1d(
            image=image,
            kernel=gauss_kern_1d,
            stride=stride_y,
            axis=0,
        ),
        kernel=gauss_kern_1d,
        stride=stride_x,
        axis=1,
    )

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

def conv1d(image, kernel, axis=-1, stride=1):
    """Performs a 1d convolution on a given axis of the input array.

    Args:
        image: A numpy array with shape (d_0,...,d_n) representing the imput image.
        kernel: The kernel to apply along the image's x-axis.
        axis: The axis to perform the convolution on.
        stride: The stride of the convolution.

    Returns:
        A numpy array with shape (d_0,...,⌈d_axis/stride⌉,...,d_n) representing the convolved image.
        Note that the input image is zero-padded to eliminate the kernel widths impact on the resulting shape.
    """
    
    axis = axis % image.ndim
    axis_width = image.shape[axis]
    w = len(kernel) // 2
    image_padded = np.pad(
        image,
        pad_width = [(w,w) if i==axis else (0,0) for i in range(image.ndim)],
        mode='edge'
    )
    image_out_shape = list(image.shape)
    image_out_shape[axis] = int(np.ceil(image_out_shape[axis] / stride))
    image_out = np.zeros(image_out_shape)
    slices = axis*(slice(None),)
    for j, x in enumerate(kernel):
        image_out += x * image_padded[slices + (slice(j, axis_width+j, stride),)]
    return image_out

def load_images(paths_to_images):
    return [load_image(path_to_image) for path_to_image in paths_to_images]

def load_image(path_to_image):
    with Image.open(path_to_image) as image_file:
        image_array = np.asarray(image_file)[:,:,:3]/255.
    return image_array

def show_image(image, title=None, block=False, out_path=None):
    plt.figure()
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.imshow(image, cmap='gray',vmin=0., vmax=1.)
    if out_path is not None:
        plt.savefig(out_path)
    plt.show(block=block)
    
def show_multiple_images(images, titles=None, block=False, out_path=None, figsize_per_image=(5.5, 5)):
    n = len(images)
    size_x, size_y = figsize_per_image
    if titles is None:
        titles = n * [None]
    fig, axs = plt.subplots(1, n, figsize=(n*size_x, size_y))
    for ax, title, image in zip(axs.reshape((-1,)), titles, images):
        if title is not None:
            ax.set_title(title)
        ax.axis('off')
        ax.imshow(image, cmap='gray',vmin=0., vmax=1.)
    if out_path is not None:
        plt.savefig(out_path)
    plt.show(block=block)

def visualize_image_pyramid(image_pyramid):
    _, global_width, channels = image_pyramid[0].shape

    images_out = []
    for image in image_pyramid:
        height, local_width, _ = image.shape
        image_out = np.zeros((height, global_width, channels))
        image_out[:, :local_width] = image
        images_out.append(image_out)
    return np.concatenate(images_out, axis=0)

def check_arrays(title, keys, arrays, arrays_true):
    print(f'Checking {title}:')
    for key, array, array_true in zip(keys, arrays, arrays_true):
        result = 'passed' if array.shape == array_true.shape and np.isclose(array, array_true).all() else 'failed'
        print(f'{key}: {result}')
    print()
