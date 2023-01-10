import numpy as np
from skimage import feature

from PIL import Image, ImageDraw
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
    
def get_gauss_kern_2d(w, sigma):
    """Returns a two-dimensional gauss kernel.

    Args:
        w: A parameter controlling the kernel size.
        sigma: The σ-parameter of the gaussian density function representing the standard deviation.

    Returns:
        A numpy array with shape (2*w+1, 2*w+1) representing a 2d gauss kernel.
        Note that array's values sum to 1.
    """
    gauss_kern = np.exp(-0.5*(np.arange(-w, w+1)**2)/sigma**2)
    gauss_kern = gauss_kern.reshape(1,-1) * gauss_kern.reshape(-1,1)
    return gauss_kern / gauss_kern.sum()

def rgb_to_gray(image_rgb):
    """Transforms an image into grayscale using reasonable weighting factors.

    Args:
        image_rgb: An rgb-image represented by a numpy array of the shape (h, w, 3).

    Returns:
        An array of the shape (h, w, 1) representing the grayscaled version of the original image.
    """
    
    # Use weighting factors presented in Lecture "bv-01-Sehen-Farbe.pdf", Slide 45
    weights = np.array([.299, .587, .114])
    
    return (image_rgb * weights).sum(axis=2, keepdims=True)

def get_colors(num_colors, key='jet'):
    cmap = plt.get_cmap(key)
    colors = cmap((np.linspace(0., 1., num_colors)))
    return [tuple([int(round(255*val)) for val in color]) for color in colors]

def draw_circles(image, radiuses, centers):
    image = Image.fromarray((255*image).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    colors = iter(get_colors(num_colors=centers[0].size))
    for radius, rows, cols in zip(radiuses, *centers):
        for row, col in zip(rows, cols):
            draw.ellipse(
                (col-radius,row-radius, col+radius, row+radius),
                outline=next(colors),
                width=1,
            )
    return np.asarray(image)/255

def canny(
    image_gray,
    sigma=np.sqrt(2),
    low_threshold=0.1,
    high_threshold=0.2
):
    """Applies the canny edge detector of skimage to a gray-scale image.

    Args:
        image_gray: A numpy array with shape (height, width, 1) representing the gray-scale imput image.
        sigma: The σ used to smooth the image before edge detection is applied.
        low_threshold: Threshold for weak edges.
        high_threshold: Threshold for strong edges.

    Returns:
        A numpy array with shape (height, width, channels) representing the filtered image.
        Note that the input image is zero-padded to preserve the original resolution.
    """
    
    return feature.canny(
        image_gray.squeeze(2),
        sigma,
        low_threshold,
        high_threshold
    )

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
    

def show_multiple_images(images, titles=None, block=False, out_path=None):
    n = len(images)
    if titles is None:
        titles = n * [None]
    fig, axs = plt.subplots(1, n, figsize=(n*(5.5),5))
    for ax, title, image in zip(axs.reshape((-1,)), titles, images):
        if title is not None:
            ax.set_title(title)
        ax.axis('off')
        ax.imshow(image, cmap='gray',vmin=0., vmax=1.)
    if out_path is not None:
        plt.savefig(out_path)
    plt.show(block=block)
    
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
    
def plot_time_and_freq(x, y_time, y_freq, title=None, out_path=None):
    n, = y_time.shape
    y_freq_shift = np.fft.fftshift(y_freq)
    
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(14,10))
    
    if title is not None:
        fig.suptitle(title, y=0.93, fontsize=14)
    
    ax_time.set_title('Time Domain')
    ax_time.set_xlabel('Time')
    ax_time.set_ylabel('Value')
    ax_time.plot(x.real, y_time.real, label='Real Part')
    ax_time.plot(x.real, y_time.imag, label='Imaginary Part')
    ax_time.legend()
    
    ax_freq.set_title('Frequency Domain')
    ax_freq.set_xlabel('Frequency')
    ax_freq.set_ylabel('Amplitude')
    
    ax_freq.bar(np.arange((1-n)//2,(n+1)//2), np.abs(y_freq_shift)/n, width=10/np.sqrt(n))
    if out_path is not None:
        plt.savefig(out_path)
    plt.show(block=False)
    
def show_time_and_freq_2d(y_time, y_freq, title=None, out_path=None):
    height, width = y_time.shape
    
    # transform amplitudes to log-scale for visualization
    y_freq_log = np.log(1 + np.abs(y_freq))
    y_freq_log_shift = np.fft.fftshift(y_freq_log)
    
    fig, (ax_time, ax_freq) = plt.subplots(1, 2, figsize=(12,6))
    
    if title is not None:
        fig.suptitle(title, y=0.93, fontsize=14)
    
    ax_time.set_title('Time Domain')
    ax_time.axis('off')
    ax_time.imshow(y_time.real, cmap='gray')
    
    ax_freq.set_title('Frequency Domain')
    ax_freq.axis('off')
    ax_freq.imshow(y_freq_log_shift, cmap='magma')
    
    if out_path is not None:
        plt.savefig(out_path)
    plt.show(block=False)
    
def check_arrays(title, keys, arrays, arrays_true):
    print(f'Checking {title}:')
    for key, array, array_true in zip(keys, arrays, arrays_true):
        result = 'passed' if np.isclose(array, array_true).all() else 'failed'
        print(f'{key}: {result}')
    print()
    
def clip(x):
    return np.clip(x, 0, 1)
