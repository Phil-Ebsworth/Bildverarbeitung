from os import path as osp
import numpy as np
from utils import show_multiple_images, clip

# Your solution starts here.
def main():
    assets = np.load('.assets.npz')
    img_boy_blurred = assets['boy_blurred']
    img_boy_corrected = reconstruct_image(
        img_boy_blurred,
        n=1 # TODO: Exercise 9a) Find and insert the filter width n
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
    return image # TODO: Exercise 9b)
# Your solution ends here.
    
if __name__ == '__main__':
    main()
