#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 09:36:06 2020

@author: Asish Gumparthi
"""

import numpy as np
import cv2
import os
import time
from matplotlib import pyplot as plt

left_file_path = "KITTI/Training/image_2/"
right_file_path = "KITTI/Training/image_3/"
disp_file_path = "KITTI/Training/disp/"
graph_file_path = "KITTI/Training/op/"

left_path_list = []
right_path_list = []
disp_path_list = []
graph_path_list = []


        
def kitti_loader():
    for i in os.listdir(left_file_path):
        filename = left_file_path + i
        left_path_list.append(filename)
    
        filename = right_file_path + i
        right_path_list.append(filename)

        filename = graph_file_path + i
        graph_path_list.append(filename)
        
        filename = disp_file_path  + i
        disp_path_list.append(filename)

    
def generate_disp(imgL, imgR):

        sad_window_size = 3;
        left_matcher = cv2.StereoSGBM_create(minDisparity= 0,
         numDisparities = 128,
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
#        start = time.time()
        displ = left_matcher.compute(imgL,imgR)
        dispr = right_matcher.compute(imgR, imgL)
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        
#        end = time.time()
#        intr = end - start
#        print(intr)
        
        
        # FILTER Parameters
        lmbda = 80000
        sigma = 1.2
        visual_multiplier = 1.0
    
        wls_filter =  cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)
        filteredImg = wls_filter.filter(displ, imgL, None, dispr) 
#        height, width = filteredImg.shape
#        filteredImg = np.delete(filteredImg, np.s_[0:50], axis=0)
#        filteredImg = np.delete(filteredImg, np.s_[height-100:height-50], axis=0)
#        filteredImg = np.delete(filteredImg, np.s_[0:50], axis=1)
#        filteredImg = np.delete(filteredImg, np.s_[width-100:width-50], axis=1)
        return filteredImg 
    

def get_accy(disparity, gt):
    tau = [3, 0.05]
    gt = np.float32(cv2.imread(gt,cv2.IMREAD_UNCHANGED))/256.0
    disparity = np.float32(disparity)/256.0
    
    index_valid = (gt != 0)
    n_total = np.sum(index_valid)

    epe = np.abs(gt - disparity)
    mag = np.abs(gt) + 1e-5

    epe = epe[index_valid]
    mag = mag[index_valid]

    err = np.logical_and((epe > tau[0]), (epe / mag) > tau[1])
    n_err = np.sum(err)

    mean_epe = np.mean(epe)
    mean_err = (float(n_err) / float(n_total))
    
    return (mean_epe, mean_err)

def iterate_disp():
#    perf_list = []

    for i in range(0, len(left_path_list)):
        print("Generating disparity for image:" + str(i))
        imgL = cv2.imread(left_path_list[i])[:,:,::-1]
        imgR = cv2.imread(right_path_list[i])[:,:,::-1]
        
#        gt_disp = cv2.imread(disp_path_list[i], cv2.IMREAD_UNCHANGED)
        
        disparity = generate_disp(imgL, imgR)
        disparity = cv2.normalize(disparity, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
        disparity = disparity.astype(np.uint16)
        
        plt.imshow(disparity, cmap = "gray")
        plt.show()
        
#        f, axarr = plt.subplots(1,2, squeeze = "True")
#        axarr[0].imshow(disparity, cmap = "gray")
#        axarr[0].title.set_text('Non scaled Image')
#        axarr[0].axis("off")
#        axarr[1].imshow(disp, cmap = "gray")
#        axarr[1].title.set_text('Scaled Image')
#        axarr[1].axis("off")
#        plt.show()
       
#        cv2.imwrite(graph_path_list[i], disparity)

        
#        f, axarr = plt.subplots(2,2,figsize=(20,8), squeeze = "True") 
#        axarr[0,0].imshow(imgL)
#        axarr[0,0].title.set_text('Left Image')
#        axarr[0,0].axis("off")
#        axarr[0,1].imshow(imgR)
#        axarr[0,1].title.set_text('Right Image')
#        axarr[0,1].axis("off")
#        axarr[1,0].imshow(disparity, "gray")
#        axarr[1,0].title.set_text('Disparity Map')
#        axarr[1,0].axis("off")
#        axarr[1,1].imshow(gt_disp, "gray")
#        axarr[1,1].title.set_text('Ground Truth')
#        axarr[1,1].axis("off")
#        plt.subplots_adjust(wspace=0, hspace=0)
#        plt.show()
#        f.savefig(graph_path_list[i])
        
        (epe,err) = get_accy(disparity, disp_path_list[i]) 
        print("performance :" + str(err)+ " EPE:" + str(epe))
#        perf_list.append(err)
#    perf_list = np.asarray(perf_list)
#    sum_avg = np.mean(perf_list)
#    print("Average Error:"+ str(sum_avg))
    
def main():
    kitti_loader()
    iterate_disp()

if __name__ == '__main__':
    main()