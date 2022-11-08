#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Authors      : Aditya Jain and Safwan Jamal
Date started : November 6, 2022
About        : Convex Optimization project; saving dissimilarity between images
"""

import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle


# In[ ]:


cifar_data_dir    = './cifar-10-images/train/'
diss_data_dir     = './dissimilarity_data/'
no_imgs_per_class = len(glob.glob('./cifar-10-images/train/airplane/*.jpg'))
# no_imgs_per_class = 100

for category in os.listdir(cifar_data_dir):    
    print(f'Calculating dissimilarity for {category} ...')
    # dissimilarity matrix and images used
    diss_matrix = np.zeros((no_imgs_per_class, no_imgs_per_class))
    image_list  = []
    
    for i in range(no_imgs_per_class):
        img_name      = str(i).zfill(4) + '.jpg'
        img_path      = cifar_data_dir + category + '/' + img_name
        img           = Image.open(img_path)
        img_grayscale = img.convert('L')
        img_array     = np.asarray(img_grayscale)
        img_i         = img_array.reshape((1, -1))        
        image_list.append(img_name)
        
        for j in range(i, no_imgs_per_class):
            img_name      = str(j).zfill(4) + '.jpg'
            img_path      = cifar_data_dir + category + '/' + img_name
            img           = Image.open(img_path)
            img_grayscale = img.convert('L')
            img_array     = np.asarray(img_grayscale)
            img_j         = img_array.reshape((1, -1))
            
            # update the dissimilarity matrix
            euclidean_dist    = np.linalg.norm(img_i-img_j)
            diss_matrix[i, j] = euclidean_dist
            diss_matrix[j, i] = euclidean_dist
     
    diss_matrix = diss_matrix/np.max(diss_matrix)
    print(f'Shape of dissimilarity matrix is {diss_matrix.shape}')
    
    with open(diss_data_dir+category+'_dissimilarity_matrix'+'.pickle', 'wb') as f:
        pickle.dump(diss_matrix, f)
        
    with open(diss_data_dir+category+'_image_list'+'.pickle', 'wb') as f:
        pickle.dump(image_list, f)
        
printf('Done with matrix calculation!')


# In[ ]:




