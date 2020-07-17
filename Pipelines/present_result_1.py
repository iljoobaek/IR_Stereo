#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:30:44 2020

@author: rtml
"""

import cv2
from matplotlib import pyplot as plt
import os

def data_loader():
    c_sgm_l = "C_SGM/left"
    c_sgm_r = "C_SGM/right"
    sgm_path = "SGM_op"
    img_right = "imgright"
    img_left = "imgleft"
    disparity_gt_path = "disp_gt"
    sgm_list = []
    c_sgm_l_path = []
    c_sgm_r_path = []
    left_path_list = []
    right_path_list = []
    disparity_list = []
    
    for image_left in os.listdir(img_left):
        a = image_left.split("t")
        l_img_path = os.path.join(img_left, image_left)
        image_right = "imgright"+a[1]
        r_img_path = os.path.join(img_right, image_right)
        left_path_list.append(l_img_path) 
        right_path_list.append(r_img_path)
        disp_path = "disp" +a[1]
        d_img_path = os.path.join(disparity_gt_path, disp_path)
        disparity_list.append(d_img_path)
    
        
    for i in range(0, len(os.listdir(c_sgm_r))):
        r_path = c_sgm_r + "/right_" +str(i) + "_disparity.png"
        l_path = c_sgm_l + "/left_" + str(i) + "_disparity.png"
        c_sgm_l_path.append(l_path)
        c_sgm_r_path.append(r_path)
        
    for i in range(0, len(os.listdir(sgm_path))):
        path = sgm_path + "/img"+str(i) + ".png"
        sgm_list.append(path)
    return left_path_list, right_path_list, disparity_list, c_sgm_l_path, c_sgm_r_path, sgm_list    

    
l_gt, r_gt, disp_gt, cl_sgm, cr_sgm, sgm_l = data_loader()
images = []
for i in range(0, 10):
        imgL = cv2.imread(l_gt[i],0)
        imgR = cv2.imread(r_gt[i],0)
        disp = cv2.imread(disp_gt[i])
        cl = cv2.imread(cl_sgm[i])
        cr = cv2.imread(cl_sgm[i])
        sg = cv2.imread(sgm_l[i])

        f, axarr = plt.subplots(3,2,figsize=(20,20)) 
        plt.subplots_adjust(wspace=0, hspace=0)
        axarr[0,0].imshow(imgL, "gray")
        axarr[0,0].title.set_text('Left Image')
        axarr[0,0].axis("off")
        axarr[0,1].imshow(imgR, "gray")
        axarr[0,1].title.set_text('Right Image')
        axarr[0,1].axis("off")
        axarr[1,0].imshow(disp, "gray")
        axarr[1,0].title.set_text('Disparity Map GT')
        axarr[1,0].axis("off")
        axarr[1,1].imshow(sg, "gray")
        axarr[1,1].title.set_text('OpenCv Map')
        axarr[1,1].axis("off")
        axarr[2,0].imshow(cl, "gray")
        axarr[2,0].title.set_text('Own SGM_left Map') 
        axarr[2,0].axis("off")
        axarr[2,1].imshow(cr, "gray")
        axarr[2,1].title.set_text('Own SGM_right Map')
        axarr[2,1].axis("off")        
        plt.show()
        op_path = "graphs/"+str(i)+".png"
        f.savefig(op_path)
graphs = "graphs"
for a in os.listdir(graphs):
    path = os.path.join(graphs,a)
    img = cv2.imread(path)
    height, width, layers = img.shape
    size = (width,height)
    images.append(img)
    
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 3, size)

for i in range(len(images)):
    out.write(images[i])
out.release()