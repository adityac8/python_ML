# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 18:59:09 2017

@author: aditya
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from time import time
# fix random seed for reproducibility
np.random.seed(7)

dataset = np.loadtxt("pima-indians-diabetes.csv",delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]

model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X,Y,epochs=150,batch_size=10)

scores = model.evaluate(X,Y)
t0=time()
print("\n%s : %.2f%%"%(model.metrics_names[1],scores[1]*100))
print "time {}".format(time()-t0)
