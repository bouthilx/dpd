#!/usr/bin/env bash

###
# First execution of all models for the zoo
##

FILENAME="$(basename "$0")"
EXECUTION_VERSION="${FILENAME%.*}"
CONTAINER="bouthilx/sgd-space-hub:495d511"


./bin/register.sh ${CONTAINER} experiments \
    --experiment l2 \
    --version ${EXECUTION_VERSION}


 ./bin/schedule.sh ${CONTAINER} experiments \
    --experiment l2 \
    --version ${EXECUTION_VERSION}
