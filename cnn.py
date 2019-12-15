# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 16:11:08 2019

@author: Matthew Carrano
"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt

record = wfdb.rdrecord('mit-bih-arrhythmia-database/102', sampto=1000, channels = [0])
anno = wfdb.rdann('mit-bih-arrhythmia-database/102', 'atr', sampto=1000)

wfdb.plot_wfdb(record=record, annotation=anno, plot_sym=True,
                   time_units='seconds', title='MIT-BIH Record 100',
                   figsize=(10,4), ecg_grids='all')
#set(anno.symbol)
# %% 
signal,fields = wfdb.rdsamp('mit-bih-arrhythmia-database/100', channels = [0])

normal = ['/']   
ids = np.in1d(anno.symbol,normal) # creates array of booleans, true means there is a normal beat

norm_beats = anno.sample[ids] # find index of normal beat peaks

norm_beats
