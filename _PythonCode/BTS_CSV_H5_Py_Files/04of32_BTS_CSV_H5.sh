#!/bin/bash
#SBATCH --account=def-curranc
#SBATCH --mem-per-cpu=16G      # increase as needed
#SBATCH --time=24:00:00
source /home/rhaghi/jupyter_py3/bin/activate
module load gcc/9.3.0 arrow python scipy-stack
pip install --no-index wetb
pip install -e /home/rhaghi/weio/.
python -c "import wetb"
python -c "import weio"
python April24_2023_TurbSimBTS_to_h5_csv_9Points_03072to04096.py