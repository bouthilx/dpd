#!/usr/bin/env bash

###
# Re-execute all models with uniformized initializations
##

FILENAME="$(basename "$0")"
EXECUTION_VERSION="${FILENAME%.*}"
CONTAINER="62351cf"


./bin/register.sh ${CONTAINER} experiments \
    --experiment whitening \
    --version ${EXECUTION_VERSION}


 ./bin/schedule.sh ${CONTAINER} experiments \
    --experiment whitening \
    --version ${EXECUTION_VERSION}
