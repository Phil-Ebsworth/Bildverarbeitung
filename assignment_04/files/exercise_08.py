from os import path as osp
import numpy as np
from utils import get_gauss_kern_2d, load_image, show_multiple_images, check_arrays, clip

# Your solution starts here.
def gauss_filter_freq(image, w, sigma):
    """Applies gauss filtering to the input image in frequency space.

    Args:
        image: A numpy array with shape (height, width, channels) representing the imput image.
        w: Defines the patch size 2*w+1 of the filter.
        sigma: The σ-parameter of the gaussian density function representing the standard deviation.

    Returns:
        A numpy array with shape (height, width, channels) representing the filtered image.
    """
    height, width = image.shape[:2]
    image_freq = np.fft.fft2(image, axes=(0,1))
    
    image_filtered_freq = image_freq # TODO: Exercise 8a)
    image_filtered_freq = np.fft.fftshift(image_filtered_freq)
    
    
    kern = get_gauss_kern_2d(w,sigma)
    H = np.fft.fft2(kern,s=(height,width))
    H = np.fft.fftshift(H)

    image_filtered_freq[:,:,0] =  np.multiply(image_filtered_freq[:,:,0] , H)
    image_filtered_freq[:,:,1] =  np.multiply(image_filtered_freq[:,:,1] , H)
    image_filtered_freq[:,:,2] =  np.multiply(image_filtered_freq[:,:,2] , H)

    image_filtered_freq = np.fft.ifftshift(image_filtered_freq)
    image_filtered = np.fft.ifft2(image_filtered_freq, axes=(0,1)) 
    return image_filtered.real
    
def inverse_gauss_filter_freq(image, w=2, sigma=0.7):
    """Performs sharpening using inverse gauss filtering in the frequency domain.

    Args:
        image: A numpy array of the shape (h,w,3) representing the image to be sharpened.
        w: Defines the patch size 2*w+1 of the filter.
        sigma: The σ-parameter of the gaussian density function representing the standard deviation.

    Returns:
        A numpy array of the shape (h,w,3) representing the sharpened image.
    """
    
    height, width = image.shape[:2]
    image_freq = np.fft.fft2(image, axes=(0,1))
    
    image_inverse_freq = image_freq # TODO: Exercise 8b)
    image_inverse_freq = np.fft.fftshift(image_inverse_freq)
         
    kern = get_gauss_kern_2d(w,sigma)
    H = np.fft.fft2(kern,s=(height,width))
    H = np.fft.fftshift(H)

    image_inverse_freq[:,:,0] =  np.divide(image_inverse_freq[:,:,0] , H)
    image_inverse_freq[:,:,1] =  np.divide(image_inverse_freq[:,:,1] , H)
    image_inverse_freq[:,:,2] =  np.divide(image_inverse_freq[:,:,2] , H)

    image_inverse_freq = np.fft.ifftshift(image_inverse_freq)
    
    image_inverse = np.fft.ifft2(image_inverse_freq, axes=(0,1)) 
    return image_inverse.real
    
def unsharp_masking(image, alpha=0.5, w=5, sigma=2.2):
    """Performs sharpening with unsharp masking on a given image.

    Args:
        image: A numpy array of the shape (h,w,3) representing the image to be sharpened.
        alpha: The weighting factor of the negative blurred image.
        w: A parameter controlling the kernel size of the gaussian blurr.
        sigma: The σ-parameter of the gaussian density function representing the standard deviation.

    Returns:
        A numpy array of the shape (h,w,3) representing the sharpened image.
    """
    image_blurred = gauss_filter_freq(image, w, sigma)
    image_sharp = image # TODO: Exercise 8c)
    image_sharp = np.add(image,np.multiply(alpha,np.subtract(image, gauss_filter_freq(image,w,sigma))))
    return image_sharp
# Your solution ends here.

def main():
    """Do not change this function at all."""

    image = load_image(osp.join('images', 'peppers.png'))
    image_blurred = clip(gauss_filter_freq(image, w=5,sigma=2.2))
    image_sharp_ig = clip(inverse_gauss_filter_freq(image, w=2, sigma=0.7))
    image_sharp_um = clip(unsharp_masking(image, alpha=0.5, w=5,sigma=2.2))
    
    show_multiple_images([image, image_blurred], ['Original', 'Blurred'])
    show_multiple_images([image_sharp_ig, image_sharp_um], ['Inverse Gauss', 'Unsharp Masking'])
    
    assets = np.load('.assets.npz')
    check_arrays(
        'Exercise 8',
        [
            'a) Gauss Filter in Frequency Domain',
            'b) Inverse Gauss Filter',
            'c) Unsharp Masking',
        ],
        [
            image_blurred,
            image_sharp_ig,
            image_sharp_um,
        ],
        [
            clip(assets['image_blurred']),
            clip(assets['image_sharp_ig']),
            clip(assets['image_sharp_um']),
        ],
    )
    input('Press ENTER to quit.')
    
if __name__ == '__main__':
    main()
