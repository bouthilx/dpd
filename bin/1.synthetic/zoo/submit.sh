#!/usr/bin/env bash

SCRIPT_PATH=$(cd $(dirname "$0") && pwd)

OMP_NUM_THREADS=1

EXECUTION_VERSION=$1
CONTAINER=$2

bash $SCRIPT_PATH/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --version ${EXECUTION_VERSION} ${@:3}


bash $SCRIPT_PATH/deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --version ${EXECUTION_VERSION} ${@:3}
