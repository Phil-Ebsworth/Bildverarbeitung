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

def compute_translation(pyramid_a, pyramid_b):
    """Computes the translation between two images using their pyramids.

    Args:
        pyramid_a: A list of numpy arrays representing the first pyramid.
        pyramid_b: A list of numpy arrays representing the second pyramid.

    Returns:
        A tuple of integers representing the translations of height and width.
    """
    a = range(-1,2)
    tuples = [(i,j) for i in a for j in a]
    y, x = 0, 0
    
    # TODO: 12b)
    levels = len(pyramid_a)

    #initial for second lowest level--------------------------------------------------------------------------------------------------------------
    height,width = 2,2
    #erstellt Tupel für alle möglichen elemente in 2x2 matrix
    x2 = range(0,2)
    tuples_x2 = [(i,j) for i in x2 for j in x2]
    # gets images of the second lowest level und fast farbkanäle zusammen
    imgA = pyramid_a[levels - 2].sum(axis=2)
    imgB = pyramid_b[levels - 2].sum(axis=2)
    # durchläuft die möglichen translationen von B
    min_mean = 300000
    for i in range(- 1, 2):
        for j in range(-1, 2):
            #definiert counter
            abs = 0
            count = 0
            #durchlaufe mögliche Tupel 
            for a in tuples_x2:
                for b in tuples_x2:
                    ''' Die Matrix A und verschobene Matrix B schneiden sich wenn:
                        x_B + i = x_A und y_B + j = y_A erfüllt sind'''
                    if(b[0] + i == a[0] & b[1] + j == a[1]):
                        abs += np.abs(np.subtract(imgA[a],imgB[b]))
                        count += 1
            #nachdem alle differenzen addiert wurden wird durch die Anzahl der treffer geteilt
            mean = abs/count  
            #sollte bei gegebener verschiebung das minimum bisheriger verschiebungen erreicht sein  
            if(mean < min_mean):
                min_mean = mean
                y,x = i,j
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #Durchlauf für restliche level--------------------------------------------------------------------------------------------------------------------------------------------
    height,width = 3,3
    x3 = range(0,3)
    #erstellt Tupel für alle möglichen elemente in 3x3 matrix
    tuples_x3 = [(i,j) for i in x3 for j in x3]

    #durchlauf der restlichen level:
    for l in range(3, levels + 1):
        #revers level order
        real_level = levels - l
        # gets images of given level und fast farbkanäle zusammen
        imgA = pyramid_a[real_level].sum(axis=2)
        imgB = pyramid_b[real_level].sum(axis=2)
        '''Da gegebenes x,y auf nächstem Level mit doppelter versetzt liegen '''
        y,x = y*2,x*2
        min_mean = 300000
        for i in range(- 2, 3):
            for j in range(-2, 3):
                #definiert counter
                abs = 0
                count = 0
                #durchlaufe mögliche Tupel 
                for a in tuples_x3:
                    for b in tuples_x3:
                        ''' Die Matrix A und verschobene Matrix B schneiden sich wenn:
                            x_B + i = x_A und y_B + j = y_A erfüllt sind'''
                        if(b[0] + i == a[0] & b[1] + j == a[1]):
                            abs += np.abs(np.subtract(imgA[a],imgB[b]))
                            count += 1
                #nachdem alle differenzen addiert wurden wird durch die Anzahl der treffer geteilt
                mean = abs/count  
                #sollte bei gegebener verschiebung das minimum bisheriger verschiebungen erreicht sein  
                if(mean < min_mean):
                    min_mean = mean
                    y,x = y+i,x+j
    return y, x

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
    print(y)
    print(x)
    
    # Compute slice limits for both images
    safezone_x, safezone_y = height_a  +  abs(y), width_a + abs(x)
    #print(safezone_y)
    
    # Create an array large enough to hold both images.
    result = np.zeros((height_a, width_a, chs)) # TODO: 12c)
    result = np.zeros((safezone_x, safezone_y, chs))


    
    # Insert images a and b according to the slices computed above.
    if (y > 0):
        if (x > 0):
            result[0:height_a,0:width_a] = image_a
            result[y:height_b+y,x:width_b+x] = image_b
        else:
            result[0:height_a,-x:width_a-x] = image_a
            result[y:height_b+y,0:width_b] = image_b
    else:
        if (x > 0):
            result[-y:height_a-y,0:width_a] = image_a
            result[0:height_b,x:width_b+x] = image_b
        else:
            result[-y:height_a-y,-x:width_a-x] = image_a  
            result[0:height_a,0:width_a] = image_b      
    
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
