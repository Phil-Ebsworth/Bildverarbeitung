from os import path as osp
import numpy as np
from matplotlib import pyplot as plt
from utils import get_gauss_kern_2d, load_image, show_image, show_multiple_images, check_arrays, clip

def box(m):
    return np.multiply(1/m,np.ones((m,m)))





""" plt.figure(figsize=(5,5))
plt.imshow(image,cmap='gray')
plt.axis('off')
plt.show() """
assets = np.load('.assets.npz')
image = assets['boy_blurred']
height, width = image.shape[:2]
image_freq = np.fft.fft2(image,axes=(0,1))
image_filtered_freq = image_freq
image_filtered_freq = np.fft.fftshift(image_filtered_freq)
kern = box(3)
H = np.fft.fft2(kern,s=(height,width))
H = np.fft.fftshift(H)
Hn = np.add(-H,1)
Htmp = Hn
Hmin1 = np.add(Htmp,1)
for i in range(1,10):
    
    Htmp = np.multiply(Htmp, Hn)
    hmin1 = np.add(Hmin1,Htmp)

    image_filtered_freq[:,:,0] =  np.multiply(image_filtered_freq[:,:,0] , Hmin1)
    image_filtered_freq[:,:,1] =  np.multiply(image_filtered_freq[:,:,1] , Hmin1)
    image_filtered_freq[:,:,2] =  np.multiply(image_filtered_freq[:,:,2] , Hmin1)

    image_filtered_freq = np.fft.ifftshift(image_filtered_freq)
    image_filtered = np.fft.ifft2(image_filtered_freq, axes=(0,1)) 
    plt.figure(figsize=(5,5))
    plt.imshow(image_filtered.real,cmap='gray')
    plt.axis('off')
    plt.show()

