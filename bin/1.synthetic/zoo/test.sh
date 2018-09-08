#!/usr/bin/env bash

CONTAINER=$1

flow-execute $CONTAINER kleio run --config /config/kleio.core/kleio_config.yaml ${@:2}
