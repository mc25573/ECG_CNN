# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 16:11:08 2019

@author: Matthew Carrano
"""

# Prepare Data
from keras.utils import to_categorical
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

def toInt(labels,charList):
    for i,beat in enumerate(charList):
        labels = np.where(labels==beat,i,labels)
    return labels.astype(int)

beat_types = ['N','V','R','L','A','E']

tst_labels = np.load('../tst_labels.npy')
tr_labels = np.load('../tr_labels.npy')

tst_labels = toInt(tst_labels,beat_types)       
tst_labels = to_categorical(tst_labels,num_classes=len(beat_types))

tr_labels = toInt(tr_labels,beat_types)       
tr_labels = to_categorical(tr_labels,num_classes=len(beat_types))
tr_labels = np.float16(tr_labels)
tst_labels = np.float16(tst_labels)

tr_im = np.load('../tr_im.npy')
tst_im = np.load('../tst_im.npy')

tr_im = tr_im.reshape(tr_im.shape[0],tr_im.shape[1],tr_im.shape[1],1) # channel last
tst_im = tst_im.reshape(tst_im.shape[0],tst_im.shape[1],tst_im.shape[1],1)

# CNN
model = Sequential()
model.add(Conv2D(32,(3,3),strides=(1,1), input_shape=(128,128,1),kernel_initializer='glorot_uniform',activation='elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),strides=(1,1),kernel_initializer='glorot_uniform',activation='elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(128,(3,3),strides=(1,1),kernel_initializer='glorot_uniform',activation='elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),strides=(1,1),kernel_initializer='glorot_uniform',activation='elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(256,(3,3),strides=(1,1),kernel_initializer='glorot_uniform',activation='elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),strides=(1,1),kernel_initializer='glorot_uniform',activation='elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(2048,activation='elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(tr_im,tr_labels,batch_size=128,epochs=5,verbose=1,shuffle=True) # shuffle is important here because the data is rather organized

score = model.evaluate(tst_im, tst_labels, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

    