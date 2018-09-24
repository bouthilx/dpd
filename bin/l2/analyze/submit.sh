#!/usr/bin/env bash

SCRIPT_PATH=$(cd $(dirname "$0") && pwd)

export OMP_NUM_THREADS=1

CONTAINER=$1
EXECUTION_VERSION=$2
ANALYSIS_VERSION=$3
ANALYSIS_CONF=$4

echo CONTAINER $CONTAINER

bash $SCRIPT_PATH/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} --analysis-version ${ANALYSIS_VERSION} \
    ${@:5}
    
bash $SCRIPT_PATH/deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --analysis-version ${ANALYSIS_VERSION}
