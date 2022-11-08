from os import path as osp
import numpy as np
from utils import gauss_function, load_image, load_images, show_image, check_arrays
import math


a = np.array([1,2,3])
b = np.array([4,5,6])

c = a+b
f = np.multiply(a,b)
d = np.vstack([a,b])
f = np.multiply(d,d)
print(f)
print(d)
print(d.shape)
e = np.zeros_like(d[1])

print(e)
for i in d:
    e = e + i

print(e)
print(e / 2)

d[1,2] = 3
e[1] = 3

print(e)
print(d)
images_tree = load_images([osp.join('images', f'tree_{i}.jpg') for i in range(5)])

print(images_tree[0].shape)
print(len(images_tree)) 

""" image_cup_noisy = load_image(osp.join('images', 'cup_noisy.png'))
print(image_cup_noisy)

print(image_cup_noisy.shape)
print(image_cup_noisy[1,1,1])
image_cup_noisy[1,1,1] = 2
result = np.zeros_like(image_cup_noisy)
result[1,1,1] = image_cup_noisy[1,1,1]
print(image_cup_noisy[1,1,1])
print(result[1,1,1])

w = 2 
print( image_cup_noisy[5-w: 5+w+1, 9-w: 9+w+1, 0])
block = image_cup_noisy[5-w: 5+w+1, 9-w: 9+w+1,0]

print(block.sum())

print(block.shape)
print(np.sort(block, axis=None))

print( image_cup_noisy[5-w: 5+w+1, 9-w: 9+w+1].shape)
print(int((2*2+1)**2 / 2))

A = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
print(gauss_function(A,2,2)) """

def g_func(x,y,sig):
    t1 = 1 / (2 *math.pi * sig**2)
    t2 = (x**2+y**2)/(2*sig**2)
    result = t1 * math.exp(- t2)
    return result

def g_func_freq(u,v,sig):
    tmp =  sig**2 * (u^2 + v^2) / 2
    tmp = tmp * math.exp(-tmp)

    return tmp

print(g_func(-1,-1,1))
print(g_func(-1,0,1))
print(g_func(-1,1,1))
print(g_func(0,-1,1))
print(g_func(0,0,1))
print(g_func(0,1,1))
print(g_func(1,-1,1))
print(g_func(1,0,1))
print(g_func(1,1,1))

print( (-1^2+1^2)/(2*1**2))

