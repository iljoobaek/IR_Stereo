#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 21:27:53 2020

@author: rtml
"""

import numpy as np
import cv2
import os

from matplotlib import pyplot as plt

op_path = "cmu/disparity_map_16/"

def CMU_data_loader():
    left_path = "CMU_dataset/left_inv_3" 
    right_path = "CMU_dataset/right_inv_3"
    
    
    same_list = []
    left_list = []
    right_list = []
    
    for a in os.listdir(left_path):
        b = a.split("_")
        c = b[5] + "_" + b[6]+ "_" + b[7]
        left_list.append(c)
    
    for c in os.listdir(right_path):
        d = c.split("_")
        e = d[5] + "_" + d[6]+ "_" + d[7]
        right_list.append(e)
        
    for i in left_list:
        if (i in right_list):
            same_list.append(i)        
    return same_list

def data_loader():
    path_list = []
    left_path = "cmu/left/" 
    for i in os.listdir(left_path):
        path_list.append(i)
    return path_list
            
def generate_disp(imgL, imgR):

        sad_window_size = 3;
        left_matcher = cv2.StereoSGBM_create(minDisparity= 0,
         numDisparities = 16,
         blockSize = 1,
         uniquenessRatio = 10,
         speckleWindowSize = 100, 
         speckleRange = 32,
         preFilterCap = 63,
         disp12MaxDiff = 1,
         P1 = sad_window_size*sad_window_size*4,
         P2 = sad_window_size*sad_window_size*32,
         mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

        displ = left_matcher.compute(imgL,imgR)
        dispr = right_matcher.compute(imgR, imgL)
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        
        
        # FILTER Parameters
        lmbda = 8000
        sigma = 1.2
        visual_multiplier = 1.0
    
        wls_filter =  cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)
        filteredImg = wls_filter.filter(displ, imgL, None, dispr) 

        return filteredImg
    
def iterate_disp(path):


    for i in range(0, len(path)):
        #i =0 
        print("Generating disparity for image:" + str(i))
        imgL = cv2.imread("cmu/left/" + path[i])[:,:,::-1]
        imgR = cv2.imread("cmu/right/" + path[i])[:,:,::-1]
        disparity = generate_disp(imgL, imgR)
 
        plt.imshow(disparity, cmap = "gray")
        plt.show()
        #cv2.waitKey(0)
   # cv2.destroyAllWindows()
#        cv2.imwrite(op_path+path[i], disparity)
        
#        disparity = cv2.normalize(disparity, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
#        disparity = disparity.astype(np.uint16)
        
#        f, axarr = plt.subplots(1,2, figsize=(10,8),squeeze = "True") 
#        plt.imshow(disparity, cmap = "gray")
#        axarr[0].imshow(disparity, cmap = "gray")
#        axarr[1].imshow(displ, cmap = "gray")
#        plt.show()
#        plt.savefig(op_path + path[i])

#        f, axarr = plt.subplots(1,3, figsize=(10,8),squeeze = "True")
#        axarr[0].imshow(imgL)
#        axarr[0].title.set_text('Left Image')
#        axarr[0].axis("off")
#        axarr[1].imshow(imgR)
#        axarr[1].title.set_text('Right Image')
#        axarr[1].axis("off")
#        axarr[2].imshow(disparity, cmap = "gray")
#        axarr[2].title.set_text('Disparity Image')
#        axarr[2].axis("off")
#        plt.show()
#        f.savefig(op_path + path[i])
        

def main():
    path = data_loader()
    iterate_disp(path)

if __name__ == '__main__':
    main()