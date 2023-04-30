# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:01:59 2022

@author: radha
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


timestr = time.strftime('%Y%m%d')
#%%
bts =[]

for file in glob.glob("../TurbSim/*.bts"):
    bts.append(file)

f_name_init = int(11)
bts.sort()
InitialOutput = weio.read(bts[0])

yz=np.array([3,7,11])

output = np.zeros((9,InitialOutput['u'].shape[1],len(bts)))

#%%
#for b in range(0,len(bts)):
df_big_csv = pd.DataFrame()
df_big_csv_1Hz = pd.DataFrame()

for b in range(0,len(bts)):
    i=0
    output_csv = np.zeros((9,InitialOutput['u'].shape[1]))
    df_output_csv = pd.DataFrame(columns=['y','z','File Name'])
    for y in yz:
        for z in yz:
            o = weio.read(bts[b])
            u = o['u']
            output[i,:,b] = u[0,:,y,z]
            output_csv[i,:]= u[0,:,y,z]
            df_output_csv.loc[i,'y'] = y
            df_output_csv.loc[i,'z'] = z
            df_output_csv.loc[i,'File Name'] = bts[b][f_name_init:]
            i=i+1
            print(y,z,bts[b])
    df = pd.DataFrame(output_csv)
    df_1Hz = pd.DataFrame(signal.resample(df,660,axis=1))
    df_output_csv_1Hz = pd.concat([df_output_csv,df_1Hz],axis=1)
    df_output_csv = pd.concat([df_output_csv,df],axis=1)
    df_big_csv = pd.concat([df_big_csv,df_output_csv],ignore_index=True)
    df_big_csv_1Hz = pd.concat([df_big_csv_1Hz,df_output_csv_1Hz],ignore_index=True)
    

#%%


h5f = h5py.File('../PostProcessing/WindX_9_GridPoints_3_7_11.h5', 'w')
h5f.create_dataset('dataset_1', data=output)
h5f.close()

h5f = h5py.File('../PostProcessing/WindX_9_GridPoints_3_7_11_1Hz.h5', 'w')
h5f.create_dataset('dataset_1', data=signal.resample(output,660,axis=1))
h5f.close()

df_big_csv.to_csv('../PostProcessing/WindX_9_GridPoints_3_7_11.csv')
df_big_csv_1Hz.to_csv('../PostProcessing/WindX_9_GridPoints_3_7_11_1Hz.csv')
