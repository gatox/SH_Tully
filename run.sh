#!/bin/bash
#SBATCH --no-requeue
#SBATCH --partition=medium
#SBATCH --job-name=qchem.in
#SBATCH --output=qchem.in.out
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=salazar.xavier@gmail.com


export QCSCRATCH=/local/qchem_${SLURM_JOB_ID} 

mkdir -p ${QCSCRATCH}

python /home/salazar/pysurf_plugins/vv_fssh_propagator.py 

