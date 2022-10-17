#!/bin/bash
#SBATCH --no-requeue
#SBATCH --partition=long
#SBATCH --job-name=qchem.in
#SBATCH --output=qchem.in.out
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=240:00:00
#SBATCH --mem-per-cpu=16G
##SBATCH --mail-type=ALL
##SBATCH --mail-user=<user_name>@gmail.com

export QCSCRATCH=/local/qchem_${SLURM_JOB_ID} 

mkdir -p ${QCSCRATCH}

export OMP_NUM_THREADS=12
python /<pysurf_folder>/pysurf/bin/run_trajectory.py 

