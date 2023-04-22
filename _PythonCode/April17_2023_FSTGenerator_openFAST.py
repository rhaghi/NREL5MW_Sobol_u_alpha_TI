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

nos = 2**6
U = cp.Uniform(lower = 3, upper=25)
#TI = cp.Uniform(lower = 0.025, upper = (0.18/U)*(6.8
 #                                     +0.75*U
 #                                     +(3*(10/U)**2))
 #               )

TI = cp.Uniform(lower = 0.04, upper=0.18*(0.75+5.6/U))



a_LB = 0.15
a_UB = 0.22
D = 126
R = D/2
U_max = 25
z = 90




alpha_lb = a_LB-0.23*(U_max/U)*(1-(0.4*np.log10(R/z))**2)
alpha_ub = a_UB + 0.4*(R/z)*(U_max/U)





alpha = cp.Uniform(lower =alpha_lb , upper=alpha_ub)


joint_dist1 = cp.J(U,alpha)
joint_dist2 = cp.J(U,TI)
joint_dist1_T = cp.Trunc(joint_dist1,lower=-0.3,upper=30)

s1 = cp.generate_samples(nos, joint_dist1_T,rule='S')
samp1 = s1[:,np.argsort(s1[0,:])]
s2 = cp.generate_samples(nos, joint_dist2,rule='S')
samp2 = s2[:,np.argsort(s2[0,:])]


# Plotting

def a_lb(U):
    if (a_LB-0.23*(U_max/U)*(1-(0.4*np.log10(R/z))**2)) < -0.3:
        lb = -0.3
    else:
        lb = a_LB-0.23*(U_max/U)*(1-(0.4*np.log10(R/z))**2) 
    return lb

U_ = np.linspace(3,25,1000)
a_lower_b = np.zeros_like(U_)
i=0

for u in U_:
    a_lower_b[i] = a_lb(u)
    i=i+1


#scale=(0.18/U_)*(6.8+0.75*U_+(3*(10/U_)**2))
scale = 0.18*(0.75+5.6/U_)
#loc = (0.025+np.zeros_like(scale))
loc = (0.04+np.zeros_like(scale))



alpha_lb = a_LB-0.23*(U_max/U_)*(1-(0.4*np.log(R/z))**2)
alpha_ub = a_UB + 0.4*(R/z)*(U_max/U_)
alpha_lb_const = -0.3+np.zeros_like(U_) 

plt.close('all')
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

marker_size = 1

ax[0].scatter(samp1[0],samp1[1],s=marker_size)
#ax[0].plot(U_[438:],alpha_lb[438:],color='r')
#ax[0].plot(U_[:438],alpha_lb_const[:438],color='r')
ax[0].plot(U_,a_lower_b,color='r')
ax[0].plot(U_,alpha_ub,color='r')
#ax[0].axvline(x=25,ymin = 0.11,ymax =0.29, color='r')
ax[0].plot((25,25),(alpha_lb[-1],alpha_ub[-1]),color='r')
#ax[0].axvline(x=3,ymin = 0.045,ymax =0.95, color='r')
ax[0].plot((3,3),(alpha_lb_const[0],alpha_ub[0]),color='r')
ax[0].set_xlabel('Wind Speed [m/s]')
ax[0].set_ylabel(r'Shear ( $\alpha$ )[-]')
ax[0].grid()






ax[1].scatter(samp2[0],samp2[1],s=marker_size)
ax[1].plot(U_,scale,color='r')
ax[1].plot(U_,loc,color='r')
#ax[1].axvline(x=3,ymin=0.045,ymax=0.95,color='r')
#ax[1].axvline(x=25,ymin=0.045,ymax=0.104,color='r')
ax[1].plot((3,3),(loc[0],scale[0]),color='r')
ax[1].plot((25,25),(loc[-1],scale[-1]),color='r')
ax[1].set_xlabel('Wind Speed [m/s]')
ax[1].set_ylabel('Turbulence Intensity [-]')
ax[1].grid()

#fig.savefig('Alpha_TI_SobolSamp_4096.png')

'''Stack alpha and TI together. In the main sampling array, the first row is the
wind speed, the second row is the shear (alpha) and the third row is TI 
'''
samp = np.vstack((samp1,samp2[1,:]))

#%%





NoOfSeeds = nos
HH = 90
TrasTime = 60
SimLength = np.array([600])
SeedNo = np.arange(1,nos+1)
DecInFN = 5 #Decimal in file naming
#%%
Sobol_Seed_Matrix = np.round(np.vstack((samp,SeedNo.T)),5)
df = pd.DataFrame(Sobol_Seed_Matrix.T,columns=['u','shear','TI','Seed No'])
df = df.astype({'Seed No': 'int32'})
df.to_csv('Sobol_Seed_Matrix.csv')
#%%

def DeciToStr(x,zfl_x,zfl_d):
    n = np.trunc(x).astype(int)
    ndci = np.abs(x-n)
    if x<0:
        n_str = 'n'+str(np.abs(n)).zfill(zfl_x-1)
    else:
        n_str = str(n).zfill(zfl_x)
    ndci_str = str(int(ndci*np.power(10,zfl_d))).zfill(zfl_d) 
    return n_str,ndci_str



#%%
TurbSimInputTemp = "../TurbSim/_Template/HH_UREFmps_SeedNo.inp"
fTurbSimInput = weio.FASTInputFile(TurbSimInputTemp)
#TurbSimExeFile = "turbsim"
#TurbSimBatFile = "../TurbSim/"+timestr+'_TurbSimBatFile.sh'
#fTurbSimBatFile = open(TurbSimBatFile,"w")
#fTurbSimBatFile.write("#!/bin/bash\n") 

for uats in zip(samp.T,SeedNo):    
    fTurbSimInput = weio.FASTInputFile(TurbSimInputTemp)
    fTurbSimInput['URef'] = uats[0][0]
    fTurbSimInput['RandSeed1'] = uats[1].astype(int)
    fTurbSimInput['NumGrid_Z'] = int(15)
    fTurbSimInput['NumGrid_Y'] = int(15)
    fTurbSimInput['AnalysisTime'] = int(SimLength+TrasTime)
    fTurbSimInput['WrADHH'] = 'False'
    fTurbSimInput['IECturbc'] = uats[0][0]
    fTurbSimInput['PLExp'] = uats[0][1]
    u_n, u_dec = DeciToStr(uats[0][0],2,DecInFN)
    TI_n, TI_dec = DeciToStr(uats[0][2],2,DecInFN)
    a_n, a_dec = DeciToStr(uats[0][1],2,DecInFN)
    TurbSimInputFileName =str(HH)+'m_'+u_n+'p'+u_dec+'mps_ti_'+TI_n+'p'+TI_dec+'_alp_'+a_n+'p'+a_dec+'_'+str(uats[1].astype(int)).zfill(6)+'.inp'
    fTurbSimInput.write("../TurbSim/"+TurbSimInputFileName)
            

#%%
InflowInputTemp = "../Inflow/_Template/NRELOffshrBsline5MW_InflowWind_XXmps.dat" 

InflowFileNameList={}

for uats in zip(samp.T,SeedNo):    
    fInflowInput = weio.FASTInputFile(InflowInputTemp)
    fInflowInput['m/s'] = uats[0][0]
    u_n, u_dec = DeciToStr(uats[0][0],2,DecInFN)
    TI_n, TI_dec = DeciToStr(uats[0][2],2,DecInFN)
    a_n, a_dec = DeciToStr(uats[0][1],2,DecInFN)
    BTSFileName =str(HH)+'m_'+u_n+'p'+u_dec+'mps_ti_'+TI_n+'p'+TI_dec+'_alp_'+a_n+'p'+a_dec+'_'+str(uats[1].astype(int)).zfill(6)+'.bts'
    fInflowInput['WindType']= 3
    fInflowInput['FileName_BTS'] = '"../TurbSim/'+BTSFileName+'"'
    InflowFileName = "NREL5MW_InflowWind_"+u_n+'p'+u_dec+'mps_ti_'+TI_n+'p'+TI_dec+'_alp_'+a_n+'p'+a_dec+'_'+str(uats[1].astype(int)).zfill(6)+'.dat'
    fInflowInput.write("../Inflow/"+InflowFileName)

#%%
        
HH = 90
#WindSpdRange = np.array([])

DLC = 'DLC12'
WindDir =0
WaveDir = 0
SimLength = 600
TrasTime = 60;
dt= 0.05/8
dt_out = 0.05

FASTTemp = "../Sims/_Template/5MW_Land_DLL_WTurb.fst"
fFASTTemp = weio.FASTInputFile(FASTTemp)
SimFolder = '../Sims/DLC12_NREL5MW'+'_'+str(WindDir).zfill(3)+'_'+str(WaveDir).zfill(3)+'_'+str(SimLength).zfill(3)+'s'

if  os.path.exists(SimFolder) == False:
    os.mkdir(SimFolder)

for uats in zip(samp.T,SeedNo): 
    fFASTTemp = weio.FASTInputFile(FASTTemp)
    u_n, u_dec = DeciToStr(uats[0][0],2,DecInFN)
    TI_n, TI_dec = DeciToStr(uats[0][2],2,DecInFN)
    a_n, a_dec = DeciToStr(uats[0][1],2,DecInFN)
    fFASTTemp['DT']=dt
    fFASTTemp['TMax']=SimLength+TrasTime
    fFASTTemp['InflowFile']='"../../Inflow/'+"NREL5MW_InflowWind_"+u_n+'p'+u_dec+'mps_ti_'+TI_n+'p'+TI_dec+'_alp_'+a_n+'p'+a_dec+'_'+str(uats[1].astype(int)).zfill(6)+'.dat"'
    fFASTTemp['TStart'] = TrasTime
    fFASTTemp['DT_Out'] = dt_out
    if uats[0][0] > 19:
        fFASTTemp['EDFile']= '"NRELOffshrBsline5MW_Onshore_ElastoDyn_Pitch20deg.dat"'
    FASTFileName = 'NREL5MW_'+DLC+'_'+u_n+'p'+u_dec+'mps_ti_'+TI_n+'p'+TI_dec+'_alp_'+a_n+'p'+a_dec+'_'+str(WindDir).zfill(3)+'_'+str(WaveDir).zfill(3)+'_'+str(uats[1].astype(int)).zfill(6)+'.fst'
    fFASTTemp.write(SimFolder+'/'+FASTFileName)



#%%

files_fst=[]

for file in glob.glob(SimFolder+'/*.fst'):
    files_fst.append(file)

files_fst.sort()


files_inp=[]

for file in glob.glob("../TurbSim/*.inp"):
    files_inp.append(file)

files_inp.sort()






#%%    
FASTExeFile = "openfast"
TurbSimExeFile = "turbsim"


NoFiles = 16
TotalNum = nos
it_steps = np.arange(0,TotalNum+1,NoFiles)
#it_steps = np.append(it_steps,nos)

m=0
#%%
for i in range(1,it_steps.size):
    FASTBatFile = '../Sims/'+str(i).zfill(4)+'of'+str(it_steps.size-1).zfill(4)+'_NREL5MW_sims'+'.sh'
    fFASTBatFile = open(FASTBatFile,"w")
    fFASTBatFile.write("#!/bin/bash\n")
    fFASTBatFile.write("#SBATCH --account=def-curranc\n")
    fFASTBatFile.write("#SBATCH --mem-per-cpu=4G      # increase as needed\n")
    fFASTBatFile.write("#SBATCH --time=24:00:00\n")
    fFASTBatFile.write("module load StdEnv/2020  intel/2020.1.217 openfast/3.1.0\n")  
    for j in range(it_steps[i-1],it_steps[i]):
        fFASTBatFile.write(TurbSimExeFile+' '+files_inp[j]+'\n')
        fFASTBatFile.write(FASTExeFile+' '+files_fst[j][8:]+'\n')
        m=m+1
    fFASTBatFile.write('rm ../TurbSim/*.sum'+ '\n')    
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
