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

ext = ['min','max','avg','std','del','del_1Hz']

if  os.path.exists('_Asmbd_Fls') == False:
    os.mkdir('_Asmbd_Fls')


for e in ext:
    csv_files = []
    df_pot = pd.DataFrame()
    for file in glob.glob('*'+e+'.csv'):
        csv_files.append(file)
    csv_files.sort()
    for f in csv_files:
        df = pd.read_csv(f)
        df_pot = pd.concat([df_pot,df],ignore_index=True)
    df_pot.to_csv('_Asmbd_Fls/NREL5MW_DLC12_Sobol_u_TI_alpha_'+e+'.csv')
    
                                                                                                                                                                                                                                                                                                                                                                                                                           
    
os.chdir(cwd)