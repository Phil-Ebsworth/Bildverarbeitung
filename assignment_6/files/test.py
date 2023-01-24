from os import path as osp
import numpy as np
#from matplotlib import pyplot as plt
from utils import load_image, show_image, show_multiple_images, visualize_image_pyramid, check_arrays
from utils import gauss_filter

a = np.array([[1,1,2,3],[3,0,5,3],[6,7,8,3],[6,7,8,3]])
""" min = np.argmin(a)
print("min = " + str(min))
x = min // 3
y = min % 3
print(x)
print(y)
print("minimum is = " + str(a[x][y])) """
print(a[1:1,1:3])