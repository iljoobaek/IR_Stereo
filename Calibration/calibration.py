#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:34:57 2020

@author: Asish Gumparthi
"""

import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.png')
#op = "op/"
i = 0
tmp = []

def pre_processing(img):
    #-----Converting image to LAB Color model----------------------------------- 
    lab= cv.cvtColor(img, cv.COLOR_BGR2LAB)
    
    
    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv.split(lab)
    
    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe1 = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe1.apply(l)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv.merge((cl,a,b))
    
    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    return final

for fname in images:
    im = cv.imread(fname,1)
    #img = pre_processing(im)
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    gray = cv.bitwise_not(gray)
    #cv.imshow('inv', gray)
    
    # Find the chess board corners
    print(fname)
    ret, corners = cv.findChessboardCorners(gray, (9,6), flags = cv.CALIB_CB_ADAPTIVE_THRESH)
    
    print(ret)
    # If found, add object points, image points (after refining them)
    if ret == True:
        i = i+1
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        color = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        cv.drawChessboardCorners(im, (9,6), corners2, ret)
        cv.drawChessboardCorners(color, (9,6), corners2, ret)
        #cv.imwrite(op + fname, im)
        cv.imshow('Original img with checkerboard overlay', im)
        cv.imshow("inverted and enchanced image", color)
        cv.waitKey(100)
        tmp.append(fname)
print("total images" + str(len(images)))
print("Recognized images" + str(i))
cv.destroyAllWindows()



# ## Calibration
# print("Calibrating...")
# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
#
## Writing camera parameters in file
#f = open("R-matrix.txt", "w")
#for i in range(0,67):
#    a = rvecs[i].T
#    f.write(tmp[i] + " : "+ str(a) + "\n")
#f.close()
#
#f = open("T-matrix.txt", "w")
#for i in range(0,67):
#    a = tvecs[i].T
#    f.write(tmp[i] + " : "+ str(a) + "\n")
#f.close()

# Un Distorting 

#img = cv.imread(images[0])
#h,  w = img.shape[:2]
#newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
#
#dst = cv.undistort(img, mtx, dist, None, newcameramtx)
## crop the image
#x, y, w, h = roi
#dst = dst[y:y+h, x:x+w]
#cv.imwrite('calibresult.png', dst)

# Re-projection Error
# print("Computing reprojeciton error")
# mean_error = 0
# for i in range(len(objpoints)):
#     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
#     mean_error += error
# print( "total error: {}".format(mean_error/len(objpoints)) )