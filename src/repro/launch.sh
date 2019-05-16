#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=186G 
#SBATCH --gres=gpu:4
#SBATCH --time=11:59:00
##SBATCH -o /dev/null # NO STDOUT
##SBATCH -e /dev/null # STDERR
##SBATCH -o /scratch/claurent/slurm/slurm.%A_%a.%N.out
##SBATCH -e /scratch/claurent/slurm/slurm.%A_%a.%N.err

# 1. Create env
source $SCRATCH/env/bin/activate

# 2. Copy data
mkdir $SLURM_TMPDIR/hpo
cp -r /scratch/claurent/data $SLURM_TMPDIR

python main.py $1 $2 
