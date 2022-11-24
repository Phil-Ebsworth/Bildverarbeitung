from os import path as osp
import numpy as np
from utils import gauss_function, load_image, show_image, check_arrays
from exercise_04 import get_gauss_kern_2d

# Your solution starts here.
def bilateral_filter(image, w, sigma_d, sigma_r):
    """Applies bilateral filtering to the input image.

    Args:
        image: A numpy array with shape (height, width, channels) representing the imput image.
        w: Defines the patch size 2*w+1 of the filter.
        sigma_d: sigma for the pixel distance
        sigma_r: sigma for the color distance

    Returns:
        A numpy array with shape (height, width, channels) representing the filtered image.
        Note that the input image is zero-padded to preserve the original resolution.
    """
    height, width, chs = image.shape
    # Pad the image corners with zeros to preserve the original resolution.
    image_padded = np.pad(image, pad_width=((w,w), (w,w), (0,0)))
    result = np.zeros_like(image)
    gauss_kern_d = get_gauss_kern_2d(w, sigma_d)[:,:,None] # TODO: Solve Exercise 4c) first!
    gauss_kern_r = get_gauss_kern_2d(w, sigma_r)[:,:,None]
    # TODO: Exercise 5) Hint: You may use gauss_function implemented in utils.py which is already imported.
    for i in range(w, height - w):
        for j in range(w, width -w):
            for c in range(chs):
                #gauss for pixel distance
                """ block_d = image[i-w: i+w+1, j-w: j+w+1,c]
                gauss_d = np.multiply(block_d,gauss_kern_d).sum()
                result[i,j,c] =  gauss_d """
                #gauss for color distance
                """ block_r = image[i-w: i+w+1, j-w: j+w+1,c]
                gauss_r = np.multiply(block_r,gauss_kern_r).sum()
                test = np.multiply(gauss_d, gauss_r)
                result[i,j,c] =  test """
    return result
# Your solution ends here.

def main(show_cup=True, show_peppers=True):
    """You may vary the parameters w and sigma to explore the effect on the resulting filtered images.
    
    Note: The test-cases will only pass with w=2, sigma_d=1.5 and sigma_r=0.15.
    """
    image_cup_noisy = load_image(osp.join('images', 'cup_noisy.png'))
    image_peppers = load_image(osp.join('images', 'peppers.png'))
    if show_cup:
        show_image(image_cup_noisy, title='Original Cup')
    if show_peppers:
        show_image(image_peppers, title='Original Peppers')

    # bilateral filter
    image_cup_bilateral_filtered = bilateral_filter(image_cup_noisy, w=2, sigma_d=1.5, sigma_r=0.15)
    image_peppers_bilateral_filtered = bilateral_filter(image_peppers, w=2, sigma_d=1.5, sigma_r=0.15)
    if show_cup:
        show_image(image_cup_bilateral_filtered, title='Bilateral-Filtered Cup')
    if show_peppers:
        show_image(image_peppers_bilateral_filtered, title='Bilateral-Filtered Peppers')
    
    assets = np.load('.assets.npz')
    check_arrays(
        'Exercise 5',
        ['a) bilateral-filtered cup', 'a) bilateral-filtered peppers',],
        [image_cup_bilateral_filtered, image_peppers_bilateral_filtered,],
        [assets['image_cup_bilateral_filtered'], assets['image_peppers_bilateral_filtered'],],
    )
    input('Press ENTER to quit.')

if __name__ == '__main__':
    main()
