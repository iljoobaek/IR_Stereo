#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:07:00 2020

@author: rtml
"""

import cv2
import matplotlib.pyplot as plt

path = "test.png"


imag = cv2.imread(path,1)

img_not = cv2.bitwise_not(imag)



def inc_ctr(img):
    #-----Converting image to LAB Color model----------------------------------- 
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    
    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    
    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe1 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe1.apply(l)
    clahe2 = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8,8))
    cl2 = clahe2.apply(l)
    
    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    limg2 = cv2.merge((cl2,a,b))
    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    final2 = cv2.cvtColor(limg2, cv2.COLOR_LAB2BGR)
    return final, final2


t1,t2 = inc_ctr(imag)
n1,n2 = inc_ctr(img_not)


cv2.imshow('orig final', t1)
cv2.imshow('orig new final', t2)
cv2.imshow('inv final', n1)
cv2.imshow('inv new final', n2)


cv2.waitKey(0)
cv2.destroyAllWindows()