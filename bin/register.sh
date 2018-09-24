#!/usr/bin/env bash

# Note: This script will execute python code inside a container

export OMP_NUM_THREADS=1

CONTAINER=$1

flow-execute ${CONTAINER} \
    python3.6 /repos/sgd-space/bin/deploy.py register ${@:2}
