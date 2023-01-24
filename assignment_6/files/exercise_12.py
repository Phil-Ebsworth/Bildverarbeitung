from os import path as osp
import numpy as np
#from matplotlib import pyplot as plt
from utils import load_image, show_image, show_multiple_images, visualize_image_pyramid, check_arrays
from utils import gauss_filter

# Your solution starts here.  
def get_gauss_pyramid(image, w, sigma):
    """Creates a gaussian image pyrimid from a given image.

    Args:
        image: A numpy array with shape (h,w,chs) representing the imput image.
        w: The width of the gauss kernel.
        sigma: The parameter controlling the bell-curve's width.

    Returns:
        A list of length ⌊log2(min(h, wi)⌋ containing numpy arrays representing the levels of a gauss pyramid.
    """
    height, width, _ = image.shape
    num_levels = int(np.log2(min(height, width)))
    gauss_pyramid = [image]
    
    # TODO: 12a)
    for i in range(num_levels):
        gauss_pyramid.append(gauss_filter(gauss_pyramid[i],w,sigma,stride_x = 2,stride_y= 2))
    
    return gauss_pyramid

def init(img1,img2):
    a = range(2)
    tuples_init = [(i,j) for i in a for j in a]
    mean = 300
    tmp = ((0,0),(0,0))
    for i in tuples_init:
        for j in tuples_init:
            ab = abs(img1[i][0] - img2[j][0])+abs(img1[i][1] - img2[j][1])+abs(img1[i][2] - img2[j][2])
            if(ab < mean):
                mean = ab
                tmp = (i,j)
    return tmp[0],tmp[1]
def compute_translation(pyramid_a, pyramid_b):
    """.

    Args:
        pyramid_a: A list of numpy arrays representing the first pyramid.
        pyramid_b: A list of numpy arrays representing the second pyramid.

    Returns:
        A tuple of integers representing the translations of height and width.
    """

    a = range(-1,2)
    tuples = [(i,j) for i in a for j in a]
    
    y, x = 0, 0
    len1 = len(pyramid_a)
    
    last_pos_a,last_pos_b = init(pyramid_a[len1-2],pyramid_b[len1-2])
    #Durchläuft die Ebenen der Pyramide
    for c in range(3,len1):
        img_a = pyramid_a[len1 - c]
        img_b = pyramid_b[len1 - c]
        #Durchläuft die 9 Punkte um de zuvor errechneten Punkt
        mean = 300
        for i in tuples:
            # pos_a ist tupel(x,y) der jetzigen Position auf höhere ebene angepasst
            pos_a = (last_pos_a[0] * 2 + i[0],last_pos_a[1] * 2 + i[1])
            for j in tuples:
                # pos_a ist tupel(x,y) der jetzigen Position auf höhere ebene angepasst
                pos_b = (last_pos_b[0] * 2 + i[0],last_pos_b[1] * 2 + i[1])
                absolut = abs(img_a[pos_a][0] - img_b[pos_b][0])+abs(img_a[pos_a][1] - img_b[pos_b][1])+abs(img_a[pos_a][0] - img_b[pos_b][1])
                if (absolut < mean):
                    mean = absolut
                    new_pos_a,new_pos_b = pos_a,pos_b
        last_pos_a,last_pos_b = new_pos_a, new_pos_b
    # TODO: 12b)
    
    x,y = last_pos_a[0] - last_pos_a[0],last_pos_a[1] - last_pos_a[1]
    return x,y

def blend_images(image_a, image_b, translation):
    """Blends two images by a given a translation.

    Args:
        image_a: A list of numpy arrays representing the first pyramid.
        image_b: A list of numpy arrays representing the second pyramid.
        translation: A tuple (y,x) denoting how image_b is shifted compared to image_a. 

    Returns:
        A numpy array holding the blended version of the two images.
        The overlapping area is the average of the respective image parts.
    """
    height_a, width_a, chs_a = image_a.shape
    height_b, width_b, chs_b = image_b.shape
    assert chs_a == chs_b
    chs = chs_a
    
    y, x = translation
    print(width_b)
    abs_height = height_a + height_b
    abs_width = width_a + width_b
    result = np.zeros((abs_height, abs_width, chs))
    result[0:height_a,0:width_a,:] = image_a
    result[y:y+height_b,x:x+width_b,:] = image_b
    # Compute slice limits for both images
    
    # Create an array large enough to hold both images.
    #result = np.zeros((height_a, width_a, chs)) # TODO: 12c)

    
    # Insert images a and b according to the slices computed above.
    
    
    # Divide the overlap by 2.

    
    return result

# Your solution ends here.

def main():
    """Do not change this function at all."""
    
    town_hall_left = load_image(osp.join('images/', 'town_hall_left.jpg'))
    town_hall_right = load_image(osp.join('images/', 'town_hall_right.jpg'))
    
    landscape_left = load_image(osp.join('images/', 'landscape_left.jpg'))
    landscape_right = load_image(osp.join('images/', 'landscape_right.jpg'))
    
    gauss_pyramid_town_hall_left = get_gauss_pyramid(town_hall_left, w=5, sigma=3.)
    gauss_pyramid_town_hall_right = get_gauss_pyramid(town_hall_right, w=5, sigma=3.)
    
    translation_town_hall = compute_translation(
        gauss_pyramid_town_hall_left,
        gauss_pyramid_town_hall_right
    )
    translation_town_hall_inv = compute_translation(
        gauss_pyramid_town_hall_right,
        gauss_pyramid_town_hall_left
    )
    
    town_hall_blended = blend_images(
        town_hall_left,
        town_hall_right,
        translation_town_hall
    )
    
    translation_landscape = (-9, 458)
    landscape_blended = blend_images(
        landscape_left,
        landscape_right,
        translation_landscape
    )
    
    show_multiple_images(
        [
            visualize_image_pyramid(gauss_pyramid_town_hall_left),
            visualize_image_pyramid(gauss_pyramid_town_hall_right),
        ],
        figsize_per_image=(3,6),
        titles=['Pyramid Left Image', 'Pyramid Right Image']
    )
    
    show_image(town_hall_blended, title='Town Hall Blended')
    show_image(landscape_blended, title='Landscape Blended')
    
    assets = np.load('.assets.npz', allow_pickle=True)
    
    check_arrays(
        'Exercise 12 a)',
        [f'Level {i}' for i in range(len(assets['gauss_pyramid_town_hall_left']))],
        assets['gauss_pyramid_town_hall_left'],
        gauss_pyramid_town_hall_left
    )
    
    check_arrays(
        'Exercise 12 b)',
        [
            'Translation',
             'Symmetry',
        ],
        [
            np.array(translation_town_hall),
            -np.array(translation_town_hall_inv),
        ],
        [
            assets['translation_town_hall'],
            assets['translation_town_hall'],
        ],
    )
    
    check_arrays(
        'Exercise 12 c)',
        ['Blending'],
        [landscape_blended],
        [assets['landscape_blended'] ],
    )

    input('Press ENTER to quit.')   
    
if __name__ == '__main__':
    main()
