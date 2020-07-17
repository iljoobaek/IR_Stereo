#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:53:57 2020

@author: rtml
"""

import numpy as np
import os
import shutil



op_left = "CMU_dataset/new_left/"
op_right = "CMU_dataset/new_right/"

def CMU_data_loader():
    left_path = "CMU_dataset/Left/" 
    right_path = "CMU_dataset/Right/"
    
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


path = CMU_data_loader()

for i in range(0, len(path)):

    in_L = "CMU_dataset/Left/Cam_L_30_05_2020_" + path[i]
    out_L = "CMU_dataset/new_left/" + path[i]
    dst = shutil.copyfile(in_L, out_L)
    
    in_R = "CMU_dataset/Right/Cam_R_30_05_2020_" + path[i]   
    out_R = "CMU_dataset/new_right/" + path[i]
    dst = shutil.copyfile(in_R, out_R)

with open("file.list", 'w') as f:
    for s in path:
        f.write(str(s) + '\n')

    