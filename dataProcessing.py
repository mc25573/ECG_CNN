# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:13:47 2019

@author: Matthew Carrano
"""

# DATA PROCESSING, training/testing data must be created separately
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import cv2

tr_rec = ['101','106','109','112','115','116','118','119','122','124','201','203','207','208','209','215','220','223'] # training records
#tst_rec = ['100','103','105','111','113','200','222','124','212','213','231','232','233'] # test records
tr_im = None
tr_labels = None

for k,num in enumerate(tr_rec):    
    sampto = 650000 # entire signal equals 650000 
    
    signal,fields = wfdb.rdsamp('mit-bih-arrhythmia-database/'+num,sampto=sampto,channels = [0]) # read signal data
    anno = wfdb.rdann('mit-bih-arrhythmia-database/'+num, 'atr',sampto=sampto) # read annotation data
    signal = np.array(signal) # convert to numpy array
       
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
    
    beat_idx = np.array(anno_idx[ids]) # find index of specific beat peaks
    
    signals = np.zeros((200,len(beat_idx)))
    
    # each column is a beat
    for i,idx in enumerate(beat_idx):
        signals[:,i] = np.array(signal[idx-100:idx+100,0]) # save 200 unit window of data centered at a beat index
     
    del signal # clear to free up memory
    
    # Convert to image    
    im_size = 128
    im_gray = np.float16(np.zeros((len(beat_idx),im_size,im_size))) # float16 to save memory    
    
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
    
    im_gray /= 255 # normalize
    
    # only concatenate if not first loop
    if tr_im is None:
        tr_im = im_gray
        tr_labels = labels
    else:
        tr_im = np.concatenate((tr_im,im_gray))
        tr_labels = np.append(tr_labels,labels)
        
del im_gray, im, labels, signals # clear mem
#np.save('tst_im.npy',tst_im)
#np.save('tst_labels.npy',tst_labels)