#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 13:44:57 2022

@author: lab
"""

import weio
#import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt
import pickle
#import seaborn as sns
import scipy.stats as st
import statistics as sts
import scipy.fftpack
import scipy.special
import scipy.io as sio
from datetime import date
import time
import pandas as pd
import glob, os, shutil


timestr = time.strftime('%Y%m%d')
#%%

df = pd.read_csv('../Sims/DLC12_NREL5MW_000_000_600s/RemSims.csv')
rem_sim = df['Rem Sims'].dropna().to_list()
rem_sim.sort()

#%%

WindDir =0
WaveDir = 0
SimLength = 600
TrasTime = 60;
dt= 0.05/16
dt_out = 0.05
TotalTime = SimLength+TrasTime
DLC = 'DLC12'

#os.mkdir('../Sims/DLC12_WindPACT1p5MW_SimLength'+str(TotalTime).zfill(3)+'s_test')
sim_folder = '../Sims/DLC12_NREL5MW'+'_'+str(WindDir).zfill(3)+'_'+str(WaveDir).zfill(3)+'_'+str(SimLength).zfill(3)+'s'
sim_folder_gen = '../Sims'
sim_folder_dlc = 'DLC12_NREL5MW'+'_'+str(WindDir).zfill(3)+'_'+str(WaveDir).zfill(3)+'_'+str(SimLength).zfill(3)+'s'
#%%
for f in rem_sim :
    fFASTTemp = weio.FASTInputFile(sim_folder+'/'+f+'.fst')
    fFASTTemp['DT']=dt
    #fFASTTemp['TMax']=TotalTime+TrasTime
    #fFASTTemp['InflowFile']='"../../Inflow/'+InflowFileName+'"'
    #fFASTTemp['TStart'] = TrasTime
    #fFASTTemp['DT_Out'] = dt_out
    fFASTTemp['EDFile'] = '"NRELOffshrBsline5MW_Onshore_ElastoDyn_Pitch20deg.dat"'
    FASTFileName = sim_folder+'/'+f+'.fst'
    fFASTTemp.write(FASTFileName)

#%%

FASTExeFile = "openfast"


NoFiles = 9
TotalNum = len(rem_sim)
it_steps = np.arange(0,TotalNum+1,NoFiles)
#it_steps = np.append(it_steps,TotalNum)

m=0
#%%
fast_sh_list=[]

for i in range(1,it_steps.size):
    FASTBatFile = sim_folder_gen+'/'+str(i).zfill(2)+'of'+str(it_steps.size-1)+'_NREL5MW_rem_sim_'+'.sh'
    fFASTBatFile = open(FASTBatFile,"w")
    fFASTBatFile.write("#!/bin/bash\n")
    fFASTBatFile.write("#SBATCH --account=def-curranc\n")
    fFASTBatFile.write("#SBATCH --mem-per-cpu=4G      # increase as needed\n")
    fFASTBatFile.write("#SBATCH --time=16:00:00\n")
    fFASTBatFile.write("module load StdEnv/2020  intel/2020.1.217 openfast/3.1.0\n")  
    for j in range(it_steps[i-1],it_steps[i]):
        fFASTBatFile.write(FASTExeFile+' '+sim_folder_dlc+'/'+rem_sim[j]+'.fst'+'\n')
        m=m+1
        print(m)
    fFASTBatFile.close()
    fast_sh_list.append(str(i).zfill(2)+'of'+str(it_steps.size-1)+'_NREL5MW_rem_sim_'+'.sh')
    
#%%

sbatch_file = sim_folder_gen+'/'+timestr+'_SHELLofshell_rem.sh'
sbatch ='sbatch'
sbatchf = open(sbatch_file,'w')


fast_sh_list.sort()

for f in fast_sh_list:
    sbatchf.write(sbatch+' '+f+'\n')

sbatchf.close()