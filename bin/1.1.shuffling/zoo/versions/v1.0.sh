#!/usr/bin/env bash

###
# First execution of all models for the zoo
##

FILENAME="$(basename "$0")"
EXECUTION_VERSION="${FILENAME%.*}"
CONTAINER="37b75ff"

./bin/1.synthetic/zoo/submit.sh ${EXECUTION_VERSION} ${CONTAINER}
