#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 07:56:38 2022

@author: lab
"""

#read through csvs and make linear regressions
#import matplotlib.pyplot as plt
import pandas as pd
#import numpy as np
#from scipy.stats import zscore
#import seaborn as sns
#from scipy import stats
import os, glob

df_dic={}
#%%

cwd = os.getcwd()
os.chdir('../Sims/DLC12_NREL5MW_000_000_600s')


files_fst = []
files_outb = []

for file in glob.glob("*.fst"):
    files_fst.append(file[:file.find('.')])

for file in glob.glob("*.outb"):
    files_outb.append(file[:file.find('.')])

diff = list(set(files_fst)-set(files_outb))

df = pd.DataFrame()

df['FAST Input'] = pd.Series(files_fst)
df['FAST Output'] = pd.Series(files_outb)
df['Rem Sims'] = pd.Series(diff)

df.to_csv('RemSims.csv',index=False)

os.chdir(cwd)