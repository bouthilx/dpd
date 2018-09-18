#!/usr/bin/env bash

export OMP_NUM_THREADS=1

CONTAINER=$1

flow-execute $CONTAINER \
    python3.6 /repos/sgd-space/bin/1.synthetic/analyze/save.py \
        --configs /repos/sgd-space/configs ${@:2}
