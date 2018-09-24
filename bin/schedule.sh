#!/usr/bin/env bash

# Note: This script is supposed to be run on the cluster, no container is used for the submission.
# But the submitted script will be executed within the container.

SCRIPT_PATH=$(cd $(dirname "$0") && pwd)

export OMP_NUM_THREADS=1

CONTAINER=$1
JOB_TYPE=$2

python ${SCRIPT_PATH}/deploy.py schedule ${JOB_TYPE} --container ${CONTAINER} ${@:3}
