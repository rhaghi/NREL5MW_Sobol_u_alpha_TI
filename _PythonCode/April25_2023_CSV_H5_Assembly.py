#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:14:20 2023

@author: lab
"""

import weio
import os
import numpy as np
#mport matplotlib.pyplot as plt
from datetime import date
import time
import pandas as pd
import glob
import h5py
from scipy import signal
#%%

cwd = os.getcwd()

os.chdir('../PostProcessing')

if  os.path.exists('_Asmbd_Fls') == False:
    os.mkdir('_Asmbd_Fls')


h5_1Hz=[]

for file in glob.glob('*_1Hz_*.h5'):
    h5_1Hz.append(file)

h5_1Hz.sort()
data_h5_1Hz=[]

for f in h5_1Hz:
    hf = h5py.File(f, 'r')
    data = hf['dataset_1'][:]
    data_h5_1Hz.append(data) 

data_h5_1Hz = np.concatenate(data_h5_1Hz, axis=2)


CSV_1Hz=[]

for file in glob.glob('*_1Hz_*.csv'):
    CSV_1Hz.append(file)

CSV_1Hz.sort()

data_csv_1Hz = pd.DataFrame()

for f in CSV_1Hz:
    df = pd.read_csv(f)
    data_csv_1Hz = pd.concat([data_csv_1Hz,df],ignore_index=True)





h5_all = []

for file in glob.glob('*.h5'):
    h5_all.append(file)
    
h5_1Hz=[]

for file in glob.glob('*_1Hz_*.h5'):
    h5_1Hz.append(file)

h5 = list(set(h5_all) - set(h5_1Hz))

h5.sort()


data_h5=[]

for f in h5:
    hf = h5py.File(f, 'r')
    data = hf['dataset_1'][:]
    data_h5.append(data) 

data_h5 = np.concatenate(data_h5, axis=2)



CSV_all = []



for file in glob.glob('*.csv'):
    CSV_all.append(file)



CSV_1Hz=[]

for file in glob.glob('*_1Hz_*.csv'):
    CSV_1Hz.append(file)


CSV = list(set(CSV_all) - set(CSV_1Hz))

CSV.sort()

data_csv = pd.DataFrame()

for f in CSV:
    df = pd.read_csv(f)
    data_csv = pd.concat([data_csv,df],ignore_index=True)



os.chdir('../_PythonCode')





#%%


h5f = h5py.File('../PostProcessing/_Asmbd_Fls/WindX_9_GridPoints_3_7_11_1Hz.h5', 'w')
h5f.create_dataset('Wind_9X', data=data_h5_1Hz)
h5f.close()

data_csv_1Hz.to_csv('../PostProcessing/_Asmbd_Fls/WindX_9_GridPoints_3_7_11_1Hz.csv')

h5f = h5py.File('../PostProcessing/_Asmbd_Fls/WindX_9_GridPoints_3_7_11.h5', 'w')
h5f.create_dataset('Wind_9X', data=data_h5)
h5f.close()

data_csv.to_csv('../PostProcessing/_Asmbd_Fls/WindX_9_GridPoints_3_7_11_1Hz.csv')



