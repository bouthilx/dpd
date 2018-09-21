#!/usr/bin/env bash

###
# First execution of all analyses, on first batch of zoo generation.
###

FILENAME="$(basename "$0")"
ANALYSIS_VERSION="a-${FILENAME%.*}"
EXECUTION_VERSION="alpha-v1.1"
CONTAINER="495d511"


# Simplified Fisher-Rao norm
ANALYSIS_CONF=configs/1.2.whitening/analyzes/fisher_rao_norm.yaml
./bin/1.2.whitening/analyze/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --analysis-version "fisher-rao-norm;${ANALYSIS_VERSION}" \
    --models mlp1wb mlp2wb mlp5wb --epochs 1,50,100,150,200,250,300


# KFAC Fisher-Rao norm
ANALYSIS_CONF=configs/1.2.whitening/analyzes/kfac_fisher_rao_norm.yaml
./bin/1.2.whitening/analyze/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --analysis-version "kfac-fisher-rao-norm;${ANALYSIS_VERSION}" \
    --epochs 1,50,100,150,200,250,300


# Participation ratio
ANALYSIS_CONF=configs/1.2.whitening/analyzes/participation_ratio.yaml
./bin/1.2.whitening/analyze/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --analysis-version "pr;${ANALYSIS_VERSION}" \
    --models mlp1 mlp2 mlp1wb mlp2wb --epochs 1,50,100,150,200,250,300


# Block Diagonal Participation ratio
ANALYSIS_CONF=configs/1.2.whitening/analyzes/bd_participation_ratio.yaml
./bin/1.2.whitening/analyze/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --analysis-version "bd-pr;${ANALYSIS_VERSION}" \
    --epochs 1,50,100,150,200,250,300


# Spectral norm
ANALYSIS_CONF=configs/1.2.whitening/analyzes/spectral_norm.yaml
./bin/1.2.whitening/analyze/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --analysis-version "spectral-norm;${ANALYSIS_VERSION}" \
    --epochs 1,50,100,150,200,250,300


# l2 norm
ANALYSIS_CONF=configs/1.2.whitening/analyzes/l2.yaml
./bin/1.2.whitening/analyze/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --analysis-version "l2-norm;${ANALYSIS_VERSION}" \
    --epochs 1,50,100,150,200,250,300


./bin/1.2.whitening/analyze/deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --analysis-version ${ANALYSIS_VERSION}
