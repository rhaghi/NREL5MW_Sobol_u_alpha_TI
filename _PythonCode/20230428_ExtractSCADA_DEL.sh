#!/bin/bash
#SBATCH --account=def-curranc
#SBATCH --mem-per-cpu=32G      # increase as needed
#SBATCH --time=120:00:00

source /home/rhaghi/jupyter_py3/bin/activate
module load gcc/9.3.0 arrow python scipy-stack
pip install --no-index wetb weio
#pip install -e /home/rhaghi/weio/.
python -c "import wetb"
python -c "import weio"

python April26_2023_ExtractSCADA_DEL.py