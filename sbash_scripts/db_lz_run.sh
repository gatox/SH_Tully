#!/bin/bash
#SBATCH --no-requeue
#SBATCH --partition=long
#SBATCH --job-name=qchem.in
#SBATCH --output=qchem.in.out
#SBATCH --mem=40000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=240:00:00
#SBATCH --mem-per-cpu=40



python /<pysurf_folder>/pysurf/bin/run_trajectory.py 

