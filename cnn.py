# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 16:11:08 2019

@author: Matthew Carrano
"""

from keras.utils import to_categorical
import numpy as np

def toInt(labels,charList):
    for i,beat in enumerate(charList):
        labels = np.where(labels==beat,i,labels)
    return labels.astype(int)

beat_types = ['N','V','R','L','A','E']

tst_labels = np.load('../tst_labels.npy')
tr_labels = np.load('../tr_labels.npy')

tst_labels = toInt(tst_labels,beat_types)       
tst_labels = to_categorical(tst_labels,num_classes=6)

tr_labels = toInt(tr_labels,beat_types)       
tr_labels = to_categorical(tr_labels,num_classes=6)
#%%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

model = Sequential()

model.add(Conv2D(32,(3,3),strides=(1,1), input_shape =(128,128,1),kernel_initializer='glorot_uniform',activation='elu'))

#model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),strides=(1,1),kernel_initializer='glorot_uniform',activation='elu'))

#model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(128,(3,3),strides=(1,1),kernel_initializer='glorot_uniform',activation='elu'))

#model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),strides=(1,1),kernel_initializer='glorot_uniform',activation='elu'))

#model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(256,(3,3),strides=(1,1),kernel_initializer='glorot_uniform',activation='elu'))

#model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Conv2D(256,(3,3),strides=(1,1),kernel_initializer='glorot_uniform',activation='elu'))

#model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())

model.add(Dense(2048,activation='elu'))

#model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(tr_im,tr_labels,batch_size=128,)


    