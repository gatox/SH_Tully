#!/bin/bash
#SBATCH --no-requeue
#SBATCH --partition=long
#SBATCH --job-name=qchem.in
#SBATCH --output=qchem.in.out
#SBATCH --mem=22G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=240:00:00
#SBATCH --mem-per-cpu=22G
##SBATCH --mail-type=ALL
##SBATCH --mail-user=<user_name>@gmail.com

export QCSCRATCH=/local/qchem_${SLURM_JOB_ID} 

mkdir -p ${QCSCRATCH}

export OMP_NUM_THREADS=8

python /<home_directory>/pysurf_plugins/vv_fssh_propagator.py 

