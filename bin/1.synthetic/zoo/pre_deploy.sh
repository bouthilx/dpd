#!/usr/bin/env bash

CONTAINER=$1

flow-execute $CONTAINER \
    python3.6 /repos/sgd-space/bin/1.synthetic/zoo/save.py \
        --configs /repos/sgd-space/configs ${@:2}
