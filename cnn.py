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

#tr_rec = ['101','106','108','109','112','114','115','116','118','119','122','124','201','203','205','207','208','209','215','220','223','230']
tr_rec = ['101','106','108','109','112','114','115','116','118','119'] # training data
tr_im = None
tr_label = None

for k,num in enumerate(tr_rec):    
    #test_num = '100'
    sampto = 10000 # entire signal equals 650000 
    
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
    beat_idx
    
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
    
    im_gray /= 255
    
    if tr_im is None:
        tr_im = im_gray
        tr_labels = labels
    else:
        tr_im = np.concatenate((tr_im,im_gray))
        
    #im_gray = im_gray.reshape(im_gray.shape[0],im_size,im_size,1)










    