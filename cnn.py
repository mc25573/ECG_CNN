# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 16:11:08 2019

@author: Matthew Carrano
"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from biosppy.signals import ecg
import cv2

record = wfdb.rdrecord('mit-bih-arrhythmia-database/100', sampto=1000, channels = [0])
anno = wfdb.rdann('mit-bih-arrhythmia-database/100', 'atr', sampto=1000)

wfdb.plot_wfdb(record=record, annotation=anno, plot_sym=True,
                   time_units='seconds', title='MIT-BIH Record 100',
                   figsize=(10,4), ecg_grids='all')
#set(anno.symbol)
#%% DATA PROCESSING
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from biosppy.signals import ecg
import cv2

#tr_rec = ['101','106','108','109','112','114','115','116','118','119','122','124','201','203','205','207','208','209','215','220','223','230'] # training records
#tst_rec = ['100','103','105','111','113','117','121','123','200','202','210','212','213','214','219','221','222','228','231','232','233'] # test records
tst_rec = ['101','106','108','109','112','114','115','116','124','201','203','205','207','208'] 
tst_im = None
tst_labels = None

for k,num in enumerate(tst_rec):    
    #test_num = '100'
    sampto = 650000 # entire signal equals 650000 
    
    signal,fields = wfdb.rdsamp('mit-bih-arrhythmia-database/'+num,sampto=sampto,channels = [0])
    anno = wfdb.rdann('mit-bih-arrhythmia-database/'+num, 'atr',sampto=sampto)
    signal = np.array(signal) # convert to numpy array
    
    set(anno.symbol)
    
    beat_types = ['N','V','R','L','A','E'] # beat types the network will classify
    ids = np.in1d(anno.symbol,beat_types) # creates array of booleans, 'True' means there is a normal beat
    labels = np.array(anno.symbol) # make numpy array of labels
    
    # remove beats starting before 100 samples and after end-100 samples
    upper = sampto - 100
    anno_idx = anno.sample
    anno_idx = anno_idx[anno.sample < upper]
    ids = ids[anno.sample < upper]
    labels = labels[anno.sample < upper]
    labels = labels[anno_idx > 100]
    ids = ids[anno_idx > 100] 
    anno_idx = anno_idx[anno_idx > 100] 
    
    labels = labels[ids] # remove "non-beat" labels
    
    #np.count_nonzero(labels=='N')
    
    beat_idx = np.array(anno_idx[ids]) # find index of specific beat peaks
    
    signals = np.zeros((200,len(beat_idx)))
    
    # each column is a beat
    for i,idx in enumerate(beat_idx):
        signals[:,i] = np.array(signal[idx-100:idx+100,0])
     
    del signal # clear to free up memory
    
    # Convert to image
    
    im_size = 128
    im_gray = np.zeros((len(beat_idx),im_size,im_size))    
    
    for i in range(len(beat_idx)):
        fig = plt.figure(frameon=False)
        plt.plot(signals[:,i]) 
        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        
        filename = 'C:/Users/Matthew/Documents/GitHub/ECG_CNN/dontcare.png' # overwrites each image
        fig.savefig(filename)
        plt.close() # so that it isn't storing every figure in memory
        im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # create image array
        im_gray[i,:,:] = cv2.resize(im, (im_size, im_size), interpolation = cv2.INTER_LANCZOS4) # increase size
        #cv2.imwrite(filename, im_gray)
        #im_gray=im_gray.reshape(1,300,300,1)
    
    im_gray /= 255 # normalize
    
    # only concatenate if not first loop
    if tst_im is None:
        tst_im = im_gray
        tst_labels = labels
    else:
        tst_im = np.concatenate((tst_im,im_gray))
        tst_labels = np.append(tst_labels,labels)
        
    #im_gray = im_gray.reshape(im_gray.shape[0],im_size,im_size,1)

del im_gray, im, labels, signals

#%%
from keras.utils import to_categorical
import numpy as np

def toInt(labels,charList):
    for i,beat in enumerate(charList):
        labels = np.where(labels==beat,i,labels)
    return labels.astype(int)

beat_types = ['N','V','R','L','A','E']

tst_labels = np.load('../tst_labels.npy')
tr_labels = np.load('../tr_labels.npy')
#%%
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




    