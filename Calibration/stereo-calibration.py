#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:34:57 2020

@author: Asish Gumparthi
"""

import numpy as np
import cv2 as cv
import glob
import os
import shutil
import argparse

# Calibration settings
parser = argparse.ArgumentParser(description='Arguments for calibration and stereo rectification')
parser.add_argument('--lpath', type=str, default='./calib_selected/IR/IR_L/*.tiff', help="left data")
parser.add_argument('--rpath', type=str, default='./calib_selected/IR/IR_R/*.tiff', help="right data")
parser.add_argument('--o_path', type=str, default='./calib_selected/IR/', help="output directory")
parser.add_argument('--IR', type=int, default=1, help="RGB/IR images - set 1 for IR")
parser.add_argument('--calib', type=int, default=1, help="perform calibration?" )
parser.add_argument('--viz', type=int, default=0, help="Visualize pattern recognition results" )
opt = parser.parse_args()

#lpath = './calib_selected/IR/IR_L/*.tiff'
#rpath = './calib_selected/IR/IR_R/*.tiff'

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


def find_checkerboard(path):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(path)
    #op = "./calib_selected/IR/op_r/"
    #recog_img = "./calib_selected/recog_images/"
    i = 0
    for fname in images:
        #a = fname.split("/")
        im = cv.imread(fname,1)
        if (opt.IR==1):
            im = pre_processing(im)
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        if (opt.IR==1):
            gray = cv.bitwise_not(gray)
        
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (9,6), flags = cv.CALIB_CB_ADAPTIVE_THRESH)
        print(fname, ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            i = i+1
            objpoints.append(objp)
            imgpoints.append(corners)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            # Draw and display the corners
            if opt.viz == 1:
                cv.drawChessboardCorners(im, (9,6), corners2, ret)
    #           cv.imwrite(op + a[-1], im)
                cv.imshow('Original img with checkerboard overlay', im)
                cv.waitKey(100)
    #        shutil.copyfile(fname, recog_img+a[3])
    cv.destroyAllWindows()
    img_shape = gray.shape[::-1]
    print("total images" + str(len(images)))
    print("Recognized images" + str(i))
  
    return objpoints, imgpoints, img_shape

## Single Camera Calibration
def intrinsic_calib(objpoints, imgpoints, img_shape):
    print("Calibrating...")
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_shape,None,None, criteria)
    return ret, mtx, dist, rvecs, tvecs
    
## Stereo Calibration
def stereo_calib(objpointsL, imgpointsL, objpointsR, imgpointsR, mtx_l, dist_l, mtx_r, dist_r,img_shape):
    # Stereo Calibration
    print("Stereo Calibration...")
    retVal, cm1, dc1, cm2, dc2, r, t, e, f = cv.stereoCalibrate(objpointsL, imgpointsL, imgpointsR,  mtx_l, dist_l, mtx_r, dist_r,img_shape, flags = cv.CALIB_USE_INTRINSIC_GUESS)
    print("Stereo re-projection error",retVal)
    R1, R2, P1, P2, Q, roi_left, roi_right = cv.stereoRectify(cm1, dc1, cm2, dc2, img_shape, r, t, flags=cv.CALIB_ZERO_DISPARITY)
    
    maplx, maply = cv.initUndistortRectifyMap(cm1, dc1, R1, P1, img_shape, cv.CV_32F)
    maprx, mapry = cv.initUndistortRectifyMap(cm2, dc2, R2, P2, img_shape, cv.CV_32F)
    
    return maplx, maply, maprx, mapry

## Re-Mapping
def re_map(maplx, maply, maprx, mapry, lpath, rpath):
    print("Re-Mapping..")
    imagesL = glob.glob(lpath)
    imagesR = glob.glob(rpath)
    
    for i in range(0, len(imagesL)):
        old_imgL = cv.imread(imagesL[i])
        old_imgR = cv.imread(imagesR[i])
        
        imgL=cv.remap(old_imgL,maplx,maply,cv.INTER_LINEAR)
        imgR=cv.remap(old_imgR,maprx,mapry,cv.INTER_LINEAR)
        
        a = imagesL[i].split("/")
        op_l_path = opt.o_path+"left_rectified/"
        op_r_path = opt.o_path+"right_rectified/"
        if os.path.exists(op_l_path)==False:
            os.makedirs(op_l_path)
        if os.path.exists(op_r_path)==False:
            os.makedirs(op_r_path)
        cv.imwrite(op_l_path + a[-1], imgL)
        a = imagesR[i].split("/")
        cv.imwrite(op_r_path + a[-1], imgR)
    print("Completed!")
 
if __name__ == '__main__': 
    objpointsL, imgpointsL, img_shape = find_checkerboard(opt.lpath)
    objpointsR, imgpointsR, img_shape = find_checkerboard(opt.rpath)
    #include flag check
    if (opt.calib ==1):
        ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = intrinsic_calib(objpointsL, imgpointsL, img_shape)
        # Re-projection Error
        print("Left Re-projection Error: ", ret_l)
        ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = intrinsic_calib(objpointsR, imgpointsR, img_shape)
        # Re-projection Error
        print("Right Re-projection Error: ", ret_r)
        maplx, maply, maprx, mapry = stereo_calib(objpointsL, imgpointsL, objpointsR, imgpointsR ,mtx_l, dist_l, mtx_r, dist_r, img_shape)
        # save files
        np.save(opt.o_path + "maplx", maplx)
        np.save(opt.o_path + "maply", maply)
        np.save(opt.o_path + "maprx", maprx)
        np.save(opt.o_path + "mapry", mapry)
        re_map(maplx, maply, maprx, mapry, opt.lpath, opt.rpath)