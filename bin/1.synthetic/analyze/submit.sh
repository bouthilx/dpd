#!/usr/bin/env bash

SCRIPT_PATH=$(cd $(dirname "$0") && pwd)

OMP_NUM_THREADS=1

EXECUTION_VERSION=$1
ANALYSIS_VERSION=$2
CONTAINER=$3
ANALYSIS_CONF=$4

bash $SCRIPT_PATH/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} --analysis-version ${ANALYSIS_VERSION} \
    ${@:5}
    
bash $SCRIPT_PATH/deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --execution-version ${EXECUTION_VERSION} --analysis-version ${ANALYSIS_VERSION}
