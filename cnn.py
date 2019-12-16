# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 16:11:08 2019

@author: Matthew Carrano
"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from biosppy.signals import ecg

record = wfdb.rdrecord('mit-bih-arrhythmia-database/102', sampto=1000, channels = [0])
anno = wfdb.rdann('mit-bih-arrhythmia-database/102', 'atr', sampto=1000)

wfdb.plot_wfdb(record=record, annotation=anno, plot_sym=True,
                   time_units='seconds', title='MIT-BIH Record 100',
                   figsize=(10,4), ecg_grids='all')
#set(anno.symbol)
#%% DATA PROCESSING
test_num = '100'
sampto = 1000 # entire signal equals 650000 

signal,fields = wfdb.rdsamp('mit-bih-arrhythmia-database/' + test_num,sampto=sampto,channels = [0])
anno = wfdb.rdann('mit-bih-arrhythmia-database/' + test_num, 'atr',sampto=sampto)

set(anno.symbol)

#%%
beat_types = ['N','V','/','R','L','A','E']   
ids = np.in1d(anno.symbol,beat_types) # creates array of booleans, true means there is a normal beat


anno_idx = anno.sample
anno_idx = anno_idx[anno.sample < sampto-100]
anno_idx = anno_idx[anno_idx > 100]

beat_idx = anno.sample[ids] # find index of specific beat peaks

beat_idx

one_d_signal = []
labels = np.array(anno.symbol) # make numpy array of labels
labels = np.transpose(labels[ids]) # remove non-beat labels


for i,idx in enumerate(beat_idx):
    one_d_signal[i] = signal[idx-100:idx+100]
    