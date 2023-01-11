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
    
    # TODO: Exercise 11a)
    #Precomputing all angles to increase the speed of the algorithm
    theta = np.arange(0,360)*np.pi/180                                              #Extracting all edge coordinates
    for idx,r in enumerate(radiuses):
        #Creating a Circle Blueprint
        bprint = np.zeros((2*(r+1),2*(r+1)))
        (m,n) = (r+1,r+1)                                                       #Finding out the center of the blueprint
        for angle in theta:
            x = int(np.round(r*np.cos(angle)))
            y = int(np.round(r*np.sin(angle)))
            bprint[m+x,n+y] = 1
        constant = np.argwhere(bprint).shape[0]
        for x,y in edge_coords:                                                       #For each edge coordinates
            #Centering the blueprint circle over the edges
            X = [x-m,x+m]                                           #Computing the extreme X values
            Y= [y-n,y+n]
            if min(X)>=0 and min(Y)>=0 and max(X)<=height and max(Y)<=width:                              #Computing the extreme Y values
                hough_space[idx,X[0]:X[1],Y[0]:Y[1]] += bprint
            
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
    for i in range(num_radiuses):
        greatest_radius = np.argpartition(hough_space[i], -num_circles_per_radius)[-num_circles_per_radius:]
        rows = greatest_radius//height
        cols= greatest_radius%height

        centers[0][i] = rows
        centers[1][i] = cols
        
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
