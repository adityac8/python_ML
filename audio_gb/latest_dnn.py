# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 19:21:44 2017

@author: vikrant
"""
import os.path
import numpy as np
import argparse
from glob import glob
import librosa
import numpy as np
import os
from keras.utils import to_categorical


#Read all files
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trainpath', default='wavs', help="Path to the WAV files used for training")
args = vars(parser.parse_args())
onepath='trainpath'
pattern = os.path.join(args[onepath], '*.wav')
#ss=[]
#if os.path.isfile("mfccX.npy") and os.path.isfile("mfccY.npy"):
if True:
#    features=[]
    class_=[]
    for wavpath in glob(pattern):
        audiofile,Y=wavpath,wavpath.split('_')[0].split('\\')[1]
#        print "{} -> {}".format(audiofile,Y)  
#        y, sr = librosa.load(audiofile)
#        ftr=librosa.feature.mfcc(y=y, sr=sr)
#        print "Before flatten",ftr.shape
#        ftr=ftr.flatten()
#        print "After flatten",ftr.shape
##        if ftr.shape[0]!=8680:ss.append(audiofile)
#        features.append(ftr)
        class_.append(Y)
#    features=np.array(features)
    labels=list(set(class_))
    labels.sort()
    num_labels=[]
    for n in class_:
        if n==labels[0]:x=0
        elif n==labels[1]:x=1
        elif n==labels[2]:x=2
        elif n==labels[3]:x=3
        elif n==labels[4]:x=4
        elif n==labels[5]:x=5
        elif n==labels[6]:x=6
        elif n==labels[7]:x=7
        elif n==labels[8]:x=8
        num_labels.append(x)
    num_labels=np.array(num_labels)
    num_y = to_categorical(num_labels,num_classes=9)
#    np.save("mfccX.npy", features)
    np.save("mfccY.npy",num_y)
else:
    features=np.load("mfccX.npy")
    num_labels=np.load("mfccY.npy")    
#print features

"""
class_=np.array(class_)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(class_)
#class_=class_.T
labels=set(class_)
"""
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout

model = Sequential()
model.add(Dense(500, activation='relu', input_dim=8680))
model.add(Dropout(0.2))
model.add(Dense(500, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(9, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#feat = [i.tolist()for i in features]
#feat = np.array(feat)
model.fit(features, num_y, epochs=10, batch_size=32)
model.predict(features, batch_size=32, verbose=0)
score = model.evaluate(features, num_y, batch_size=128)
print "Accuracy",score
