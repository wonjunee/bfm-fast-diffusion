#!/bin/bash
#SBATCH -c 1  # Number of Cores per Task
#SBATCH --mem=20000  # Requested Memory
#SBATCH -t 4:00:00  # Job time limit

cd /work/pi_wuzhexu_umass_edu/CPP

module load python/3.8.5
module load anaconda/2022.10
module load gcc/5.5.0
module load fftw/3.3.8

conda activate tg3

python data_generation.py -p gaussian -ns 2 -nx 64 -nt 10 -dt 0.0005 -alp 4 -tau 10