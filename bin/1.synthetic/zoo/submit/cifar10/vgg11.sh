#!/usr/bin/env bash

#SBATCH --job-name=synthetic.cifar10.vgg1
#SBATCH --time=02:59:00

#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --export=ALL
#SBATCH --mem=32000M
#SBATCH --account=rpp-bengioy
