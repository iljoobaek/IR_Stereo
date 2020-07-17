#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:42:37 2020

@author: rtml
"""

import cv2


img  = cv2.imread("test.png")


cv2.imshow('test', img)
cv2.waitKey(0)  
  
#closing all open windows  
cv2.destroyAllWindows()