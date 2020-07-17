#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:33:41 2020

@author: rtml
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os



def find_feature_points(image_a, image_b):
    sift = cv.xfeatures2d.SIFT_create()
    # detect feature points in each image
    kp_a, des_a = sift.detectAndCompute(image_a, None)
    kp_b, des_b = sift.detectAndCompute(image_b, None)

    # Match feature points together
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_b, des_a, k=2)
    matches2 = matches

    # Filter out outliers
    filter_fn = lambda m: m[0].distance < 0.85 * m[1].distance
    matched = filter(filter_fn, matches)
    matched2 =  filter(filter_fn, matches2)

    image_a_points = np.float32(
        [kp_a[m.trainIdx].pt for (m, _) in matched]).reshape(-1, 1, 2)
    image_b_points = np.float32(
        [kp_b[m.queryIdx].pt for (m, _) in matched2]).reshape(-1, 1, 2)

    # find fundamental mat
    return image_a_points, image_b_points

def rectify_pair(image_left, image_right, viz=False):
    """Computes the pair's fundamental matrix and rectifying homographies.

    Arguments:
      image_left, image_right: 3-channel images making up a stereo pair.

    Returns:
      F: the fundamental matrix relating epipolar geometry between the pair.
      H_left, H_right: homographies that warp the left and right image so
        their epipolar lines are corresponding rows.
    """

    image_a_points, image_b_points = find_feature_points(image_left,
                                                         image_right)

    f_mat, mask = cv.findFundamentalMat(image_a_points,
                                         image_b_points,
                                         cv.RANSAC)
    imsize = (image_right.shape[1], image_right.shape[0])
    image_a_points = image_a_points[mask.ravel() == 1]
    image_b_points = image_b_points[mask.ravel() == 1]

    _, H1, H2 = cv.stereoRectifyUncalibrated(image_a_points,
                                              image_b_points,
                                              f_mat, imsize)

    return f_mat, H1, H2


l_path = "cmu/image_2/"
r_path = "cmu/image_3/"
ol_path = "cmu/left_1/"
or_path = "cmu/right_1/"
a = 0
for i in os.listdir(l_path):
    a = a+1
    print(a)
    img_l = cv.imread(l_path + i)
    img_r = cv.imread(r_path + i)

    F, H_left, H_right = rectify_pair(img_l, img_r)
    
    

    left = cv.warpPerspective(img_l,H_left,(640,512))
    right = cv.warpPerspective(img_r,H_right,(640,512))
    cv.imwrite(ol_path+i, left)
    cv.imwrite(or_path+i, right)

    b = i.split(".png")
    outtxt = "cmu/calibration_files/"+ b[0] + ".txt"

    f = open(outtxt, "w")
    f.write("H_L:" + str(H_left) + "\nH_R" + str(H_right) + "\nF:" + str(F))
    f.close()


plt.imshow(left, cmap= 'gray')
plt.imshow(right, cmap='gray')

cv.imshow("left", left)
cv.waitKey(0)
cv.destroyAllWindows()

