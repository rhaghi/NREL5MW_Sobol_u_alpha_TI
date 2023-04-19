#!/bin/bash
#SBATCH --account=def-curranc
#SBATCH --mem-per-cpu=8G      # increase as needed
#SBATCH --time=48:00:00
module load StdEnv/2020  intel/2020.1.217 openfast/3.1.0
