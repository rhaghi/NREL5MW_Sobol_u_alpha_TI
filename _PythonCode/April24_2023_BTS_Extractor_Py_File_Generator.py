#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:17:42 2023

@author: lab
"""
import numpy as np
import os
import glob

with open('April24_2023_TurbSimBTS_to_h5_csv_9Points_Template.py','r') as f:
    code_org = f.read().splitlines()
#%%
cwd = os.getcwd()
NoFiles = 1024
TotalNum = 2**15
it_steps = np.arange(0,TotalNum+1,NoFiles)
BTStoCSV='BTS_CSV_H5_Py_Files'

if  os.path.exists(BTStoCSV) == False:
    os.mkdir(BTStoCSV)

for i in range(1,it_steps.size):
    code = code_org.copy()
    code[20] = code[20] + ' ' + str(it_steps[i-1])
    code[21] = code[21] + ' ' + str(it_steps[i])
    nfi = str(it_steps[i-1]).zfill(5) + 'to' + str(it_steps[i]).zfill(5)
    with open(BTStoCSV+'/'+'April24_2023_TurbSimBTS_to_h5_csv_9Points_'+nfi+'.py','w') as f:
        for l in code:
            f.write("%s\n" % l)
#%%
os.chdir('BTS_CSV_H5_Py_Files')

py=[]

for file in glob.glob("*.py"):
    py.append(file)
    
py.sort()

for i in range(1,len(py)+1):
    py_sh_file= str(i).zfill(2)+'of'+str(it_steps.size-1)+'_BTS_CSV_H5'+'.sh'
    fpy_sh_file = open(py_sh_file,"w")
    fpy_sh_file.write("#!/bin/bash\n")
    fpy_sh_file.write("#SBATCH --account=def-curranc\n")
    fpy_sh_file.write("#SBATCH --mem-per-cpu=16G      # increase as needed\n")
    fpy_sh_file.write("#SBATCH --time=24:00:00\n")
    fpy_sh_file.write("source /home/rhaghi/jupyter_py3/bin/activate\n")
    fpy_sh_file.write("module load gcc/9.3.0 arrow python scipy-stack\n")
    fpy_sh_file.write("pip install --no-index wetb\n")
    fpy_sh_file.write('pip install -e /home/rhaghi/weio/.\n')
    fpy_sh_file.write('python -c "import wetb"\n')
    fpy_sh_file.write('python -c "import weio"\n')
    fpy_sh_file.write("python "+ py[i-1])
    fpy_sh_file.close()



files=[]

for file in glob.glob("*.sh"):
    files.append(file)
    
    
files.sort()

sbatch_file = 'SHELLofshell_py.sh'
sbatch ='sbatch'
sbatchf = open(sbatch_file,'w')



for f in files:
    sbatchf.write(sbatch+' '+f+'\n')

sbatchf.close()




os.chdir(cwd)
    
    
    