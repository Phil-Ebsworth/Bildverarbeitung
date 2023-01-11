from os import path as osp
import numpy as np
from matplotlib import pyplot as plt
from utils import load_image, show_image, rgb_to_gray, canny, draw_circles
#from exercise_10 import get_zero_crossing

# Your solution starts here.  
def circular_hough_transform(edges, radiuses):
    """Applies a Hough transform that responds to circles of given radiuses.
    
    Args:
        edges: A boolean numpy array with shape (height, width, 1)
               True pixels indicate the presence of an edge.
        radiuses: A list of radiuses the Hough space should constructed for.

    Returns:
        A numpy array with shape (len(radiuses), height, width) representing the resulting Hough space.
        Each value counts the number of edges located on a circle with its corresponding radius.
    """
    height, width = edges.shape[:2]
    num_circles = len(radiuses)
    
    hough_space = np.zeros((num_circles, height, width))
    edge_coords = np.stack(np.where(edges), axis=1)
    
    for idx, k in enumerate(radiuses):
        for u in (k, width - k):
            for v in (k, height -k):
                #if there is an edge at current pixel than add to the hough space. Pixels with max will be circle center
                if edges[u,v]:
                    hough_space[idx,u,v] += 1 
                    

    # TODO: Exercise 11a)
            
    return hough_space
    
def detect_circles(hough_space, num_circles_per_radius=1):
    """Finds the most prominent circle centers in each level of a given Hough space.
    
    Args:
        hough_space: A numpy array with shape (n, height, width) representing the input Hough space.
        num_circles_per_radius: The numbers of circle centers to return per Hough space level.

    Returns:
        A tuple of two numpy integer-arrays each of the shape (n, num_circles_per_radius).
        The first/second array contains the row/column-coordinates of the detected circle centers.
    """
    num_radiuses, height, width = hough_space.shape
    hough_space = hough_space.reshape(num_radiuses, height*width)
    
    centers = (
        np.zeros((num_radiuses, num_circles_per_radius)),
        np.zeros((num_radiuses, num_circles_per_radius))
    ) # TODO: Exercise 11b)
    return centers
# Your solution ends here.

def main():
    """Do not change this function at all."""
    
    coins = load_image(osp.join('images', 'coins.jpg'))
    coins_gray = rgb_to_gray(coins)
    
    # Detect edges using Canny.
    coins_canny = canny(coins_gray, sigma=2)
    
    radiuses=[46, 48, 52, 56, 66, 74]
    circular_hough_space = circular_hough_transform(coins_canny, radiuses)
    centers = detect_circles(circular_hough_space, num_circles_per_radius=5)
    coins_with_circles =  draw_circles(coins, radiuses, centers)
    
    # Show Results.
    for radius, hough_layer in zip(radiuses, circular_hough_space):
        plt.figure()
        plt.title(f'radius={radius}')
        plt.imshow(hough_layer)
        plt.show(block=False)
        
    show_image(coins_with_circles)

    input('Press ENTER to quit.')   
    
if __name__ == '__main__':
    main()
