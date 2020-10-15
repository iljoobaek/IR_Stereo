import os
import sys
import numpy as np
import cv2 as cv

image1_pth = sys.argv[1]
image2_pth = sys.argv[2]

alpha = 0
beta = 0
left = 0
right = 640
top = 0
btm = 512
s = 1
new_r = 848
new_c = 512
change_flag = 0

title_window = 'Linear Blend'

def translate_x(val):
    global left
    global right
    global change_flag
    if change_flag ==0:
        print(val)
        left = val
        right = left+640


def translate_y(val):
    global top
    global btm
    global change_flag
    if change_flag ==0:
        top = val
        btm = top + 512


def alpha_trans(val):
    global alpha
    global beta
    global change_flag
    if change_flag ==0:
        alpha = val / 100
        beta = ( 1.0 - alpha )

def scale(val):
    global s
    global change_flag
    global new_r
    global new_c
    global left
    global right
    global top
    global btm
    if change_flag ==0:
        s = s if s > 0 else 1
        s = val/10
        new_r, new_c = int(s*848), int(s*512)
        left = 0
        right = 640
        top = 0
        btm = 512
        print("new--",new_r, new_c)
        

def make_sliders():
    cv.namedWindow(title_window)
    cv.createTrackbar('Alpha', title_window, 0, 100, alpha_trans)
    cv.createTrackbar('X', title_window, 0, 1000, translate_x)
    cv.createTrackbar('Y', title_window, 0, 800, translate_y)
    cv.createTrackbar('Scale', title_window, 10, 20, scale)

def main():
    global change_flag
    make_sliders()
    src1 = cv.imread(cv.samples.findFile(image1_pth))
    src2 = cv.imread(cv.samples.findFile(image2_pth))
    print(src2.shape)
    tmp = image2_pth.split("/")
    while(1):
        change_flag = 1
        img1 = cv.resize(src1, (new_r, new_c), interpolation=cv.INTER_LINEAR)
        print("Orig- ",img1.shape)
        roi = img1[top:btm, left:right]
        print("ROI- ",roi.shape)
        dst = cv.addWeighted(roi, alpha, src2, beta, 0.0)
        change_flag = 0
        cv.imshow(title_window, dst)
        c = cv.waitKey(10)
        if (c == ord('s')):
            cv.imwrite("op/"+tmp[1], roi)
if __name__ == "__main__":
    main()