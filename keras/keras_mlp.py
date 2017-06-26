# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:33:48 2017

@author: aditya
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras
from keras.callbacks import ReduceLROnPlateau

# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(15, size=(1000, 1)), num_classes=15)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(15, size=(100, 1)), num_classes=15)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.

"""Input Layer""" 
model.add(Dense(50, activation='relu', input_dim=20))
model.add(Dropout(0.2))

"""Hidden Layer""" 
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))

"""Output Layer""" 
model.add(Dense(15, activation='softmax'))

#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])





"""
Without Patience"""
model.fit(x_train, y_train,
          epochs=200,
          batch_size=128)

"""For Patience
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
              patience=10, min_lr=0.001)

model.fit(x_train, y_train,
          epochs=200,
          batch_size=128, callbacks=[reduce_lr])
"""


score = model.evaluate(x_test, y_test, batch_size=128)