from os import path as osp
import numpy as np
from utils import load_image, show_multiple_images, rgb_to_gray, canny, check_arrays


# Your solution starts here.
def sobel_x_filter(image_gray):
    """Applies the horizontal sobel filter to the input image.

    Args:
        image_gray: A numpy array with shape (height, width, 1) representing the imput image.

    Returns:
        A numpy array with shape (height, width, 1) representing the horizontal sobel filtered image.
        Note that the input image is zero-padded to preserve the original resolution.
    """
    #define sobel x
    sobel_x = [ [-1,0,1],
                [-2,0,2],
                [-1,0,1]]
    #new image
    height, width = image_gray.shape[:2]
    H = np.zeros((height, width), dtype=np.float32)

    #loop over image_grey, user sobel_x and write value in new image. Try except to pass over border cases
    for u in range(height):
        for v in range(width):
            try:
                H[u][v] = (sobel_x[0][0] * image_gray[u-1][v-1] + sobel_x[0][1] * image_gray[u][v-1] + sobel_x[0][2] * image_gray[u+1][v-1] + 
                            sobel_x[1][0] * image_gray[u-1][v] + sobel_x[1][1] * image_gray[u][v] + sobel_x[1][2] * image_gray[u+1][v] + 
                            sobel_x[2][0] * image_gray[u-1][v+1] + sobel_x[2][1] * image_gray[u][v+1] + sobel_x[2][2] * image_gray[u+1][v+1])
            except:
                continue
    return H # TODO: Exercise 10a)

def sobel_y_filter(image_gray):
    """Applies the vertical sobel filter to the input image.

    Args:
        image_gray: A numpy array with shape (height, width, 1) representing the imput image.

    Returns:
        A numpy array with shape (height, width, 1) representing the vertical sobel filtered image.
        Note that the input image is zero-padded to preserve the original resolution.
    """
    #define sobel y
    sobel_y = [ [-1,-2,-1],
                [0,0,0],
                [1,2,1]]
    #new image
    height, width = image_gray.shape[:2]
    H = np.zeros((height, width), dtype=np.float32)

    #loop over image_grey, user sobel_y and write value in new image. Try except to pass over border cases
    for u in range(height):
        for v in range(width):
            try:
                H[u][v] = (sobel_y[0][0] * image_gray[u-1][v-1] + sobel_y[0][1] * image_gray[u][v-1] + sobel_y[0][2] * image_gray[u+1][v-1] + 
                            sobel_y[1][0] * image_gray[u-1][v] + sobel_y[1][1] * image_gray[u][v] + sobel_y[1][2] * image_gray[u+1][v] + 
                            sobel_y[2][0] * image_gray[u-1][v+1] + sobel_y[2][1] * image_gray[u][v+1] + sobel_y[2][2] * image_gray[u+1][v+1])
            except:
                continue
    return image_gray # TODO: Exercise 10a)

def sobel_combine(sobel_x, sobel_y):
    """Combines the vertical and the horizontal sobel filtered images.

    Args:
        sobel_x: A numpy array with shape (height, width, 1) representing the horizontal sobel filtered image.
        sobel_y: A numpy array with shape (height, width, 1) representing the vertical sobel filtered image.

    Returns:
        A numpy array with shape (height, width, 1) representing the combined sobel edges.
        Note that the input image is zero-padded to preserve the original resolution.
    """
    H = np.sqrt(np.matmul(np.power(sobel_x,2),np.power(sobel_y,2)))
    return H # TODO: Exercise 10a)

def laplace_filter(image, w, sigma_low, sigma_high):
    """Approximates a Laplacian of Gaussian filter to the input image using the difference of Gaussians.

    Args:
        image: A numpy array with shape (height, width, channels) representing the imput image.
        w: Defines the patch size 2*w+1 of the filter.
        sigma_low: The σ-parameter of the narrow bell curve.
        sigma_high: The σ-parameter of the wide bell curve.

    Returns:
        A numpy array with shape (height, width, channels) representing the filtered image.
    """
    assert sigma_low <= sigma_high

    # I don't get why we have two sigmas...
    sigma = sigma_high
    height, width = image.shape[:2]

    kernel_size = 2*w+1
    x,y = np.meshgrid(np.arange(-kernel_size/2+1,kernel_size/2+1),np.arange(-kernel_size/2+1,kernel_size/2+1))
    
    normal = 1 / (2.0 * np.pi * sigma**2)

    kernel = ((x**2 + y**2 - (2.0*sigma**2)) / sigma**4) * np.exp(-(x**2 + y**2) / (2.0*sigma**2)) / normal
    
    I = np.zeros_like(image, dtype=float)

    for u in range(width - (kernel_size -1)):
        for v in range(height - (kernel_size -1)):
            try:
                window = image[u:u+kernel_size, v:v+kernel_size] * kernel
                I[u,v] = np.sum(window)
            except:
                continue


    return I # TODO: Exercise 10b)
    
def get_zero_crossing(image_laplace):
    """Finds zero-crossing within the laplacian of an image.

    Args:
        image_laplace: A numpy array with shape (height, width, 1) representing the laplacian of an image.

    Returns:
        A boolean numpy array with shape (height, width, 1).
        A True pixel indicates that image_laplace has (at least) one zero-crossing next to its location.
    """
    height, width = image_laplace.shape[:2]
    zero_crossing = np.zeros_like(image_laplace, dtype=bool)
    
    for u in range(width):
        for v in range(height):
            if image_laplace[u][v] == 0:
                if ((image_laplace[u][v-1] < 0 and image_laplace[u][v+1] > 0) 
                or (image_laplace[u][v-1] < 0 and image_laplace[u][v+1] < 0) 
                or (image_laplace[u-1][v] < 0 and image_laplace[u+1][v] > 0) 
                or (image_laplace[u-1][v] > 0 and image_laplace[u+1][v] < 0)):
                    zero_crossing[u][v] = True
            if image_laplace[u][v] < 0:
                if ((image_laplace[u][v-1] > 0) 
                or (image_laplace[u][v+1] > 0) 
                or (image_laplace[u-1][v] > 0) 
                or (image_laplace[u+1][v] > 0)):
                    zero_crossing[u][v] = True

    return zero_crossing # TODO: Exercise 10c)
# Your solution ends here.

def main():
    """Do not change this function at all."""
    coins = load_image(osp.join('images', 'coins.jpg'))
    peppers = load_image(osp.join('images', 'peppers.png'))
    
    peppers_gray = rgb_to_gray(peppers)
    coins_gray = rgb_to_gray(coins)
    
    # Detect Edges using Sobel.
    coins_sobel_x = sobel_x_filter(coins_gray)
    coins_sobel_y = sobel_y_filter(coins_gray)
    coins_sobel = sobel_combine(coins_sobel_x, coins_sobel_y)
    
    show_multiple_images(
        [coins, coins_sobel_x, coins_sobel_y, coins_sobel],
        titles=['Original', 'Horizontal Sobel', 'Vertical Sobel', 'Combined Sobel']
    )
    
    # Detect edges using Marr-Hildreth.
    coins_laplace = laplace_filter(coins_gray, 16, sigma_low=3, sigma_high=6)
    coins_marr_hildreth = get_zero_crossing(coins_laplace)
    
    peppers_laplace = laplace_filter(peppers_gray, 16, sigma_low=3, sigma_high=6)
    peppers_marr_hildreth = get_zero_crossing(peppers_laplace)
    
    # Detect edges using Canny.
    peppers_canny = canny(peppers_gray)
    coins_canny = canny(coins_gray, sigma=2)
    
    # Compare Results.
    show_multiple_images(
        [coins, coins_marr_hildreth, coins_canny],
        ['Original', 'Marr-Hildreth', 'Canny']
    )
    
    show_multiple_images(
        [peppers, peppers_marr_hildreth, peppers_canny],
        ['Original', 'Marr-Hildreth', 'Canny']
    )
    
    # Test functions.
    assets = np.load('.assets.npz')
    check_arrays(
        'Exercise 10',
        keys=[
            'a) x-Sobel',
            'a) y-Sobel',
            'a) Combined Sobel',
            'b) Laplace Coins',
            'b) Laplace Peppers',
            'c) Marr-Hildreth Coins',
            'c) Marr-Hildreth Peppers',
        ],
        arrays=[
            coins_sobel_x,
            coins_sobel_y,
            coins_sobel,
            coins_laplace,
            peppers_laplace,
            coins_marr_hildreth,
            peppers_marr_hildreth,
        ],
        arrays_true=[
            assets['coins_sobel_x'],
            assets['coins_sobel_y'],
            assets['coins_sobel'],
            assets['coins_laplace'],
            assets['peppers_laplace'],
            assets['coins_marr_hildreth'],
            assets['peppers_marr_hildreth'],
        ]
    )
    input('Press ENTER to quit.')
    
if __name__ == '__main__':
    main()
