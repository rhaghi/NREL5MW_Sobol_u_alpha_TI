#!/bin/bash
#SBATCH --account=def-curranc
#SBATCH --mem-per-cpu=32G      # increase as needed
#SBATCH --time=24:00:00
source /home/rhaghi/jupyter_py3/bin/activate
module load gcc arrow python scipy-stack
pip install wetb
pip install weio
python April28_2023_Ext_SCADA_DEL_07168to08192.py