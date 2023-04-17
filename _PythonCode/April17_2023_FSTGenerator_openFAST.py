# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:41:42 2020

@author: rhaghi
"""

import weio
import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
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


"""
This section modifies a TurbSim template file to the wind speeds
that are required. 6 Seeds from the seed pool are selected.
"""
NoOfSeeds = 512
HH = 90
TrasTime = 60
WindSpdRange = np.arange(3,26)
SimLength = np.array([600])
RndSeedInt = np.random.choice(range(100,20000),NoOfSeeds*WindSpdRange.shape[0],replace=False) 
RandomSeedsMat = RndSeedInt.reshape((WindSpdRange.size,NoOfSeeds,SimLength.size))


#%%
TurbSimInputTemp = "../TurbSim/_Template/HH_UREFmps_SeedNo.inp"
fTurbSimInput = weio.FASTInputFile(TurbSimInputTemp)
#TurbSimExeFile = "turbsim"
#TurbSimBatFile = "../TurbSim/"+timestr+'_TurbSimBatFile.sh'
#fTurbSimBatFile = open(TurbSimBatFile,"w")
#fTurbSimBatFile.write("#!/bin/bash\n") 

#%%
for l in range(0,SimLength.size):
    k=0
    for i in WindSpdRange:
        fTurbSimInput = weio.FASTInputFile(TurbSimInputTemp)
        fTurbSimInput['URef'] = i
        RandomSeeds = RandomSeedsMat[k,:,l]
        k=k+1
        for j in RandomSeeds:
            fTurbSimInput['RandSeed1'] = j.astype(int)
            fTurbSimInput['NumGrid_Z'] = int(15)
            fTurbSimInput['NumGrid_Y'] = int(15)
            fTurbSimInput['AnalysisTime'] = int(SimLength[l]+TrasTime)
            fTurbSimInput['WrADHH'] = 'False'
            #fTurbSimInput['PLExp'] = 0
            TurbSimInputFileName =str(HH)+'m_'+str(i).zfill(2)+'mps_'+str(j.astype(int)).zfill(5)+'.inp'
            fTurbSimInput.write("../TurbSim/"+TurbSimInputFileName)
            #fTurbSimBatFile.write(TurbSimExeFile+' '+TurbSimInputFileName+'\n')

#fTurbSimBatFile.close()
#os.startfile(TurbSimBatFile)

#%%
InflowInputTemp = "../Inflow/_Template/NRELOffshrBsline5MW_InflowWind_XXmps.dat" 

InflowFileNameList={}
for i in WindSpdRange:
    fInflowInput = weio.FASTInputFile(InflowInputTemp)
    fInflowInput['m/s'] = i
    for j in RandomSeeds:
        fInflowInput = weio.FASTInputFile(InflowInputTemp)
        BTSFileName =str(HH)+'m_'+str(i).zfill(2)+'mps_'+str(j.astype(int)).zfill(5)+'.bts'
        #HHFileName =str(HH)+'m_'+str(i).zfill(2)+'mps_'+str(j.astype(int))+'.hh'
        fInflowInput.data[4]['value'] = 3
        #fInflowInput.data[15]['value'] = '"../TurbSim/'+HHFileName+'"'
        fInflowInput.data[20]['value'] = '"../TurbSim/'+BTSFileName+'"'
        InflowFileName = "NRELOffshrBsline5MW_InflowWind_"+str(i).zfill(2)+'mps_'+str(j.astype(int)).zfill(5)+'.dat'
        fInflowInput.write("../Inflow/"+InflowFileName)

#%%
        
HH = 90
#WindSpdRange = np.array([])

DLC = 'DLC12'
WindDir =0
WaveDir = 0
SimLength = np.array([600])
TrasTime = 60;
dt= 0.05/8
dt_out = 0.05

FASTTemp = "../Sims/_Template/5MW_Land_DLL_WTurb.fst"
fFASTTemp = weio.FASTInputFile(FASTTemp)
#FASTExeFile = "openfast"
#FASTBatFile = "../Sims/"+timestr+'_FASTBatFile.sh'
#fFASTBatFile = open(FASTBatFile,"w")
#fFASTBatFile.write("#!/bin/bash\n")
#fFASTBatFile.write("#SBATCH --account=def-curranc\n")
#fFASTBatFile.write("#SBATCH --mem-per-cpu=24G      # increase as needed\n")
#fFASTBatFile.write("#SBATCH --time=480:00:00\n")
#fFASTBatFile.write("module load StdEnv/2020  intel/2020.1.217 openfast/3.1.0\n")  

m=0;

for i in SimLength:
    #os.mkdir('../Sims/DLC12_NREL5MWOnShore_SimLength'+str(i.astype(int)).zfill(3)+'s')
    SimFolder = '../Sims/DLC12_NREL5MWOnShore_SimLength'+str(i.astype(int)).zfill(3)+'s'
    #SimFolder_Linux = '../Sims/DLC12_NREL5MWOnShore_SimLength'+str(i.astype(int)).zfill(3)+'s'
    #FASTBatFileFolder = SimFolder+"/"+timestr+'_FASTBatFile_DLC12_'+str(i.astype(int))+'s'+'.sh'
    #fFASTBatFileFolder = open(FASTBatFileFolder,"w")
    #fFASTBatFileFolder.write("#!/bin/bash\n")
    #fFASTBatFileFolder.write("#SBATCH --account=def-curranc\n")
    #fFASTBatFileFolder.write("#SBATCH --mem-per-cpu=24G      # increase as needed\n")
    #fFASTBatFileFolder.write("#SBATCH --time=480:00:00\n")
    #fFASTBatFileFolder.write("module load StdEnv/2020  intel/2020.1.217 openfast/3.1.0\n")                         
    l=0
    for j in WindSpdRange:
        RandomSeeds = RandomSeedsMat[l,:,m]
        l=l+1
        for k in RandomSeeds:
            fFASTTemp = weio.FASTInputFile(FASTTemp)
            fFASTTemp['DT']=dt
            fFASTTemp['TMax']=i+TrasTime
            #fFASTTemp['EDFile']= '"../../'+fFASTTemp['EDFile'][1:]
            #fFASTTemp['BDBldFile(1)']='"../../'+fFASTTemp['BDBldFile(1)'][1:]
            #fFASTTemp['BDBldFile(2)']='"../../'+fFASTTemp['BDBldFile(2)'][1:]
            #fFASTTemp['BDBldFile(3)']='"../../'+fFASTTemp['BDBldFile(3)'][1:]
            fFASTTemp['InflowFile']='"../../Inflow/'+'NRELOffshrBsline5MW_InflowWind_'+str(j).zfill(2)+'mps_'+str(k.astype(int)).zfill(5)+'.dat"'
            #fFASTTemp['AeroFile']='"../../'+fFASTTemp['AeroFile'][1:]
            #fFASTTemp['ServoFile']='"../../'+fFASTTemp['ServoFile'][1:]
            fFASTTemp['TStart'] = TrasTime
            fFASTTemp['DT_Out'] = dt_out
            FASTFileName = 'NREL5MWOnShore_'+DLC+'_'+str(j).zfill(2)+'mps_'+str(WindDir).zfill(3)+'_'+str(WindDir).zfill(3)+'_'+str(i)+'s_'+str(k).zfill(5)+'.fst'
            fFASTTemp.write(SimFolder+'/'+FASTFileName)
            #fFASTBatFile.write(FASTExeFile+' '+SimFolder_Linux+'/'+FASTFileName+'\n')
            #fFASTBatFileFolder.write(FASTExeFile+' '+FASTFileName+'\n')
    #fFASTBatFileFolder.close()
    m=m+1

#fFASTBatFile.close()



#FASTTempFolder = "../Sims/_Template/"
#dat_files = glob.iglob(os.path.join(FASTTempFolder, "*.dat"))
#for file in dat_files:
#    if os.path.isfile(file):
#        shutil.copy2(file, SimFolder)
#%%

files_fst=[]

for file in glob.glob("../Sims/DLC12_NREL5MWOnShore_SimLength600s/*.fst"):
    files_fst.append(file)

files_fst.sort()


files_inp=[]

for file in glob.glob("../TurbSim/*.inp"):
    files_inp.append(file)

files_inp.sort()






#%%    
FASTExeFile = "openfast"
TurbSimExeFile = "turbsim"


NoFiles = 12
TotalNum = 23*512
it_steps = np.arange(0,TotalNum+1,NoFiles)
it_steps = np.append(it_steps,23*512)

m=0
#%%
for i in range(1,it_steps.size):
    FASTBatFile = '../Sims/'+str(i).zfill(3)+'of'+str(it_steps.size-1)+'_NREL5MW_sims'+'.sh'
    fFASTBatFile = open(FASTBatFile,"w")
    fFASTBatFile.write("#!/bin/bash\n")
    fFASTBatFile.write("#SBATCH --account=def-curranc\n")
    fFASTBatFile.write("#SBATCH --mem-per-cpu=8G      # increase as needed\n")
    fFASTBatFile.write("#SBATCH --time=48:00:00\n")
    fFASTBatFile.write("module load StdEnv/2020  intel/2020.1.217 openfast/3.1.0\n")  
    for j in range(it_steps[i-1],it_steps[i]):
        fFASTBatFile.write(TurbSimExeFile+' '+files_inp[j]+'\n')
        m=m+1
        print(m)
    for j in range(it_steps[i-1],it_steps[i]):
        fFASTBatFile.write(FASTExeFile+' '+files_fst[j][8:]+'\n')
        m=m+1
        print(m)
    fFASTBatFile.close()                                      

#%%

os.chdir('../Sims/')

files=[]

for file in glob.glob("*.sh"):
    files.append(file)
    
    
files.sort()

sbatch_file = timestr+'_SHELLofshell.sh'
sbatch ='sbatch'
sbatchf = open(sbatch_file,'w')



for f in files:
    sbatchf.write(sbatch+' '+f+'\n')

sbatchf.close()

os.chdir('../_PythonCode')    
