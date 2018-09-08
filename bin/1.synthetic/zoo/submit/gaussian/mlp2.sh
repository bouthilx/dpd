#!/usr/bin/env bash

#SBATCH --job-name=synthetic.gaussian.mlp2.overwrite-suffix
#SBATCH --output=overwrite-me
#SBATCH --error=overwrite-me
#SBATCH --time=02:59:00

#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --export=ALL
#SBATCH --mem=32000M
#SBATCH --account=overwrite-me
#SBATCH --array=1-10
