#!/usr/bin/env bash

CONTAINER=$1

flow-execute $CONTAINER \
    python3.6 /repos/sgd-space/bin/0.benchmark/zoo/save.py \
        --configs /repos/sgd-space/configs ${@:2}
