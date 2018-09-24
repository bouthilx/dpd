#!/usr/bin/env bash

###
# Re-execute all models with uniformized initializations
##

FILENAME="$(basename "$0")"
EXECUTION_VERSION="${FILENAME%.*}"
CONTAINER="bouthilx/sgd-space-hub:350a24c"


./bin/register.sh ${CONTAINER} experiments \
    --experiment shuffling \
    --version ${EXECUTION_VERSION}


 ./bin/schedule.sh ${CONTAINER} experiments \
    --experiment shuffling \
    --version ${EXECUTION_VERSION}
