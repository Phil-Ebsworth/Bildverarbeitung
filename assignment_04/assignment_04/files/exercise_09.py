from os import path as osp
import numpy as np
from utils import show_multiple_images, clip

# Your solution starts here.
def main():
    assets = np.load('.assets.npz')
    img_boy_blurred = assets['boy_blurred']
    freq_image = np.fft.fft2(np.fft.fftshift(img_boy_blurred), axes=(0,1))
    img_boy_corrected = reconstruct_image(
        img_boy_blurred,
        n=64 # TODO: Exercise 9a) Find and insert the filter width n
    ) 

    show_multiple_images([img_boy_blurred, clip(img_boy_corrected)], ['Blurred', 'Your Reconstruction'])
    input('Press ENTER to quit.')
    
def reconstruct_image(image, n):
    """
    Applies inverted filtering to reconstruct an image that was blurred by a horizontal box-kernel.

    Args:
        image: A numpy array with shape (height, width, channels) representing the altered image.
        n: The horizontal box-kernel's width.

    Returns:
        A numpy array with shape (height, width, channels) representing the reconstructed image.
    """
    image_freq = np.fft.fft2(np.fft.fftshift(image), axes=(0,1))
    height, width = image.shape[:2]
         
    img_kern = np.zeros((height,width))
    h_off = round((height)/2)
    w_off = round((width-n)/2)
    img_kern[h_off,w_off:w_off+n] = np.ones((1,n))/n
    img_kern = np.fft.fftshift(img_kern)
    H = np.fft.fft2(img_kern)

    image_freq[:,:,0] =  np.divide(image_freq[:,:,0] , H, out=np.zeros_like(image_freq[:,:,0]), where=H!=0)
    image_freq[:,:,1] =  np.divide(image_freq[:,:,1] , H, out=np.zeros_like(image_freq[:,:,1]), where=H!=0)
    image_freq[:,:,2] =  np.divide(image_freq[:,:,2] , H, out=np.zeros_like(image_freq[:,:,2]), where=H!=0)
    
    image = np.fft.ifft2(image_freq, axes=(0,1))
    image = np.fft.ifftshift(image).real
    return image # TODO: Exercise 9b)
# Your solution ends here.
    
if __name__ == '__main__':
    main()
