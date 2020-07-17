#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:16:51 2020

@author: rtml
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def get_accy(disparity, gt):
    tau = [3, 0.05]

    gt = np.float32(cv2.imread(gt,0))
    gt = gt/256.0
    disparity = np.float32(cv2.imread(disparity,0))
    disparity = disparity/256.0
    
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

gt_path = "Output/disp/"
disp_path = "Output/right_disp/"

def data_loader(gt_path, disp_path):
    error_list = []
    for i in os.listdir(gt_path):
        a = i.split(".")
        disparity = disp_path + a[0] + "_disp.pgm"
        gt = gt_path + i
        error,_ = get_accy(disparity, gt)
        error_list.append(error)
        
    error_list = np.asarray(error_list)
    sum_avg = np.mean(error_list)
    print(sum_avg)
    return error_list

err = data_loader(gt_path, disp_path)
