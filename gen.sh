#!/bin/bash
#SBATCH -c 1  # Number of Cores per Task
#SBATCH --mem=20000  # Requested Memory
#SBATCH -t 4:00:00  # Job time limit



 

python data_generation.py -p gaussian -ns 2 -nx 64 -nt 4 -dt 0.001 -alp 4 -tau 10