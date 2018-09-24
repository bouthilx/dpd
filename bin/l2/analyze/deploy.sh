#!/usr/bin/env bash

export OMP_NUM_THREADS=1

CONTAINER=$1

# TODO: Make sbatch executable from within the container
# https://groups.google.com/a/lbl.gov/forum/#!topic/singularity/syLcsIWWzdo

python bin/1.3.l2/analyze/deploy.py --container $CONTAINER --configs configs ${@:2}

# --version alpha-v2.0 --datasets gaussian mnist cifar10 \
# --models mlp2 mlp5 lenet vgg11 resnet18 --print-only
