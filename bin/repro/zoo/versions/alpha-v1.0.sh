#!/usr/bin/env bash

###
# Execution of Mobile, VGG, ResNet and PreActResNet
###

FILENAME="$(basename "$0")"
EXECUTION_VERSION="a-${FILENAME%.*}"
CONTAINER="bouthilx/repro-hub:TODO"

VALUES="seed=[i for i in range(0, 20)]"


./bin/register.sh $CONTAINER experiments \
    --experiment repro \
    --version ${EXECUTION_VERSION} \
    --models mobilenet vgg11 resnet18 preactresnet18 \
    --values "$VALUES"


./bin/schedule.sh ${CONTAINER} experiments \
    --experiment repro \
    --version ${EXECUTION_VERSION}
