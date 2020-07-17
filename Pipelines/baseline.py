#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:14:29 2020

@author: rtml
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("data.csv")


min_distance = list(data.iloc[:,1])
err_mindist = list(data.iloc[:,2])
dat = np.asarray(data)
plt.figure(figsize=(40,30))
for i in range(0,len(min_distance)):
    dist = [min_distance[i], 0.5, 1, 2, 5, 10, 20 ,50]
    err = [err_mindist[i], dat[i,3],dat[i,4],dat[i,5],dat[i,6],dat[i,7],dat[i,8],dat[i,9]]
    a = plt.plot(dist, err)


plt.savefig("op.png")