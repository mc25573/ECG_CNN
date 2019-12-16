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
test_num = '100'
sampto = 650000 # entire signal equals 650000 

signal,fields = wfdb.rdsamp('mit-bih-arrhythmia-database/' + test_num,sampto=sampto,channels = [0])
anno = wfdb.rdann('mit-bih-arrhythmia-database/' + test_num, 'atr',sampto=sampto)
signal = np.array(signal) # convert to numpy array

set(anno.symbol)

#%%
beat_types = ['N','V','/','R','L','A','E'] # beat types the network will classify
ids = np.in1d(anno.symbol,beat_types) # creates array of booleans, 'True' means there is a normal beat
labels = np.array(anno.symbol) # make numpy array of labels

# remove data out of window range
upper = sampto - 100
anno_idx = anno.sample
anno_idx = anno_idx[anno.sample < upper]
ids = ids[anno.sample < upper]
labels = labels[anno.sample < upper]
labels = labels[anno_idx > 100]
ids = ids[anno_idx > 100] 
anno_idx = anno_idx[anno_idx > 100] 

labels = np.transpose(labels[ids]) # remove non-beat labels

beat_idx = np.array(anno_idx[ids]) # find index of specific beat peaks
beat_idx

signals = np.zeros((200,len(beat_idx)))

# each column is a beat
for i,idx in enumerate(beat_idx):
    signals[:,i] = np.array(signal[idx-100:idx+100,0])
    
#%% Convert to image
    
#newSignals = np.zeros((200,9))
#
#test= np.unique(np.sort(signals))
#for i in range(9):
#    for j in range(200):
#        newSignals[j,i] = int(np.where(test == signals[j,i])[0])
#
#
#x_train = 

fig = plt.figure(frameon=False)
plt.plot(signals[:,1]) 
plt.xticks([]), plt.yticks([])
for spine in plt.gca().spines.values():
    spine.set_visible(False)

filename = 'C:/Users/Matthew/Documents/GitHub/ECG_CNN'+ '/' + str(1)+'.png'
fig.savefig(filename)
im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
im_gray = cv2.resize(im_gray, (300, 300), interpolation = cv2.INTER_LANCZOS4)





    