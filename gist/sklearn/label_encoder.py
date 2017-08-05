# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 13:43:19 2017

@author: adityac8
"""

test_y=['a','b','a','b','c','c','a','c']
pred=  ['c','b','a','c','a','c','a','c']


from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
le2 = LabelEncoder()
le1.fit(test_y)
le2.fit(pred)

x1=le1.transform(test_y)
x2=le1.transform(pred)
print x1
print x2
