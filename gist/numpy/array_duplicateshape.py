# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 10:47:28 2017

@author: aditya
"""
import numpy as np
a=[[1],[2],[3],[4],[5]]

b=[[6],[7],[8]]

b=np.array(b)
a=np.array(a)
newb=np.zeros(a.shape)
j=0
for i in range(len(newb)):
    if i in range(len(b)):
        newb[i]=b[i]
    else:
        tmpi=b[j]
        newb[i]=tmpi
        j+=1
        
print newb
