# -*- coding: utf-8 -*-
"""
Created on Fri Sep 01 17:54:13 2017

@author: aditya
This file calculates the number of unique labels present in a datset with the help of meta file
"""

import glob
import numpy as np
from sklearn.preprocessing import LabelEncoder

for f in glob.glob("*.txt"):
    arr1=[]
    x=open(f,'r')
    print '\n===============',f
    for row in x.readlines():
        arr1.append(row.split()[1])
    print "Total Samples",len(arr1)
    uni_arr=list(set(arr1))
    uni_arr.sort()
    y=len(uni_arr)
    zeros=np.zeros(y)
    le1 = LabelEncoder()
    le1.fit(arr1)
    x1=le1.transform(arr1)
    for z in x1:
        zeros[z]+=1
        
    for i in range(y):
        print "{} = {}".format(uni_arr[i],zeros[i])
        
    print "Maximum Samples of {} which are {}".format(uni_arr[np.argmax(zeros)],np.max(zeros))
    print "Minimum Samples of {} which are {}".format(uni_arr[np.argmin(zeros)],np.min(zeros))
