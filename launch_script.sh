#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=230000MB 
#SBATCH --account=cin_staff
#SBATCH --partition=m100_usr_prod
#SBATCH --time=24:00:00
#SBATCH --error=job.%j.err
#SBATCH --output=job.%j.out
#SBATCH --job-name=nome_job

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load autoload profile/global 
module load qiskit

echo "python main.py"


