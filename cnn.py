# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 16:11:08 2019

@author: Matthew Carrano
"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt

record = wfdb.rdrecord('mit-bih-arrhythmia-database/100', sampto=1000, channels = [0])
annotation = wfdb.rdann('mit-bih-arrhythmia-database/100', 'atr', sampto=1000)



wfdb.plot_wfdb(record=record, annotation=annotation, plot_sym=True,
                   time_units='seconds', title='MIT-BIH Record 100',
                   figsize=(10,4), ecg_grids='all')

data = wfdb.rdsamp('mit-bih-arrhythmia-database/100', sampto=1000, channels = [0])