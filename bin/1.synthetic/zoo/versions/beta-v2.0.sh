#!/usr/bin/env bash

###
# First execution of all models for the zoo
##

FILENAME="$(basename "$0")"
EXECUTION_VERSION="${FILENAME%.*}"
CONTAINER="62351cf"

./bin/1.synthetic/zoo/submit.sh ${EXECUTION_VERSION} ${CONTAINER} \
    --model mlp1 mlp2 mlp5 \
    --dataset gaussian mnist cifar10

./bin/1.synthetic/zoo/submit.sh ${EXECUTION_VERSION} ${CONTAINER} \
    --model mlp1wb mlp2wb mlp5wb \
    --dataset gaussian mnist cifar10

./bin/1.synthetic/zoo/submit.sh ${EXECUTION_VERSION} ${CONTAINER} \
    --model lenet vgg11 resnet18 \
    --dataset mnist cifar10
