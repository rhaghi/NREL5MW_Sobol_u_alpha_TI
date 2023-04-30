#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 16:56:47 2022

@author: lab
"""

#from wetb.fatigue_tools.fatigue import eq_load
import weio
import os
import glob
#import fnmatch
#import chaospy as cp
import numpy as np
#import matplotlib.pyplot as plt
#import pickle
#import seaborn as sns
#import scipy.stats as st
#import statistics as sts
#import scipy.fftpack
#import scipy.special
#import scipy.io as sio
#from datetime import date
import time
import pandas as pd
import scipy.signal as signal
from pyFAST.input_output import FASTOutputFile
from pyFAST.postpro import equivalent_load
import sys




timestr = time.strftime('%Y%m%d')

cwd = os.getcwd()

start = time.time()

#%%
channels = ['Time_[s]',
 'Wind1VelX_[m/s]',
 'Wind1VelY_[m/s]',
 'Wind1VelZ_[m/s]',
 'Azimuth_[deg]',
 'RotSpeed_[rpm]',
 'GenSpeed_[rpm]',
 'RootFxb1_[kN]',
 'RootFyb1_[kN]',
 'RootFzb1_[kN]',
 'RootMxb1_[kN-m]',
 'RootMyb1_[kN-m]',
 'RootMzb1_[kN-m]',
 'RotTorq_[kN-m]',
 'YawBrFxp_[kN]',
 'YawBrFyp_[kN]',
 'YawBrMxp_[kN-m]',
 'YawBrMyp_[kN-m]',
 'TwrBsFxt_[kN]',
 'TwrBsFyt_[kN]',
 'TwrBsFzt_[kN]',
 'TwrBsMxt_[kN-m]',
 'TwrBsMyt_[kN-m]',
 'TwrBsMzt_[kN-m]',
 'NcIMUTAxs_[m/s^2]',
 'NcIMUTAys_[m/s^2]',
 'NcIMUTAzs_[m/s^2]',
 'YawBrTAxp_[m/s^2]',
 'YawBrTAyp_[m/s^2]',
 'YawBrTAzp_[m/s^2]',
 'LSSTipMys_[kN-m]',
 'LSSTipMzs_[kN-m]',
 'LSShftFys_[kN]',
 'LSShftFzs_[kN]',
 'NacYaw_[deg]',
 'RotThrust_[kN]',
 'GenPwr_[kW]',
 'GenTq_[kN-m]']
#%%
ext = ['min','max','avg','std','del','del_1Hz']

df_empty = pd.DataFrame()
df_stats= dict.fromkeys(ext,df_empty)


outb_files = []
os.chdir('../../Sims/DLC12_NREL5MW_000_000_600s')


for file in glob.glob("*.outb"):
    outb_files.append(file)    

outb_files.sort()
#outb_files=outb_files[:5]

begin = 31744
end =  32768

#%%
def NREL5MWoutbDataExtractor(filename,channels,no_bins=46,m=4,neq=600):
    df1 = weio.FASTOutputFile(filename).toDataFrame()
    df = df1.T.drop_duplicates().T
    ext = ['min','max','avg','std','del','del_1Hz']
    d_ext={}
    for e in ext:
        d_ext[e] = pd.DataFrame()
        d_ext[e]['Filename'] =pd.Series(filename)
        #d_ext[e]['Sim Wind Speed (m/s)'] =pd.Series(float(filename[15:17]+'.'+filename[18:22]))
        #d_ext[e]['Sim Std (-)'] = pd.Series(float(filename[30:30+filename[30:].find('_')])/10000)
        for c in channels:
            if e == 'min':
                d_ext[e][c] = pd.Series(df[c].min())
            if e == 'max':
                d_ext[e][c] = pd.Series(df[c].max())
            if e == 'avg':
                d_ext[e][c] = pd.Series(df[c].mean())
            if e == 'std':
                d_ext[e][c] = pd.Series(df[c].std())
            if e == 'del':
                d_ext[e][c] = equivalent_load(df[c].to_numpy(),nBins=46,m=4,Teq=600,method='rainflow_windap')
            if e == 'del_1Hz':
                ch_1Hz = signal.resample(df[c].to_numpy(),600)
                d_ext[e][c] = equivalent_load(ch_1Hz,nBins=46,m=4,Teq=600,method='rainflow_windap') 
    return d_ext

def NREL5MWoutbDataAssmble(d_ext,d_ext_append):
    ext = ['min','max','avg','std','del','del_1Hz']
    for e in ext:
        d_ext[e] = pd.concat([d_ext[e], d_ext_append[e]],ignore_index=True,axis=0)
    return d_ext

j=1

for f in outb_files[begin:end]:
    sys.stdout.write(f+'\n')
    df_ext_stat = NREL5MWoutbDataExtractor(filename = f,channels = channels, no_bins = 46, m=4, neq=600)
    df_stats = NREL5MWoutbDataAssmble(df_stats,df_ext_stat)
    sys.stdout.write('file '+ str(j) +" out of "+str(len(outb_files))+'\n')
    j=j+1

#%%
bins = np.arange(2.5,26.5,1)
labels = np.arange(1,bins.size)

os.chdir('../../PostProcessing')
nfi = str(begin).zfill(5)+'to'+str(end).zfill(5)


for e in ext:
    #df_stats[e] = df_stats[e].sort_values(by=['Sim Wind Speed (m/s)'],ignore_index=True)
    #df_stats[e]['bin number'] = pd.cut(df_stats[e]['Sim Wind Speed (m/s)'], bins=bins, labels=labels)
    df_stats[e].to_csv(nfi+'NREL5MW_DLC12_Sobol_u_TI_alpha_'+e+'.csv')

os.chdir(cwd)

end = time.time()
sys.stdout.write(str(end-start)+'\n')
