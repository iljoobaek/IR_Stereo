#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:39:53 2020

@author: rtml
"""

import numpy as np
import os 
import cv2
import matplotlib.pyplot as plt


l_path = "CMU/training/image_2/"
r_path = "CMU/training/image_3/"
o_path = "result/"
f_path = "result_graphs/"


for i in os.listdir(l_path):
    img_l = cv2.imread(l_path + i)
    img_r = cv2.imread(r_path + i)
    img_o = cv2.imread(o_path + i)
    
    f, axarr = plt.subplots(1,3, figsize=(10,8),squeeze = "True")
    axarr[0].imshow(img_l, cmap = "gray")
    axarr[0].title.set_text('Left Image')
    axarr[0].axis("off")
    axarr[1].imshow(img_r, cmap = "gray")
    axarr[1].title.set_text('Right Image')
    axarr[1].axis("off")
    axarr[2].imshow(img_o, cmap = "gray")
    axarr[2].title.set_text('Disparity Image')
    axarr[2].axis("off")
    f.savefig(f_path + i)

 