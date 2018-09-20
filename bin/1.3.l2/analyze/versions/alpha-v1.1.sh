#!/usr/bin/env bash

###
# Fix kleio bug to keep proper tags in branched children
# Add spectral norm configuration file
###

FILENAME="$(basename "$0")"
ANALYSIS_VERSION="a-${FILENAME%.*}"
EXECUTION_VERSION="alpha-v1.1"
CONTAINER="b1b8a3a"


# Simplified Fisher-Rao norm
ANALYSIS_CONF=configs/1.3.l2/analyzes/fisher_rao_norm.yaml
./bin/1.3.l2/analyze/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --analysis-version "fisher-rao-norm;${ANALYSIS_VERSION}" \
    --models mlp1wb mlp2wb mlp5wb --epochs 300


# KFAC Fisher-Rao norm
ANALYSIS_CONF=configs/1.3.l2/analyzes/kfac_fisher_rao_norm.yaml
./bin/1.3.l2/analyze/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --analysis-version "kfac-fisher-rao-norm;${ANALYSIS_VERSION}" \
    --epochs 300


# Participation ratio
ANALYSIS_CONF=configs/1.3.l2/analyzes/participation_ratio.yaml
./bin/1.3.l2/analyze/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --analysis-version "pr;${ANALYSIS_VERSION}" \
    --models mlp1 mlp2 mlp1wb mlp2wb --epochs 300


# Block Diagonal Participation ratio
ANALYSIS_CONF=configs/1.3.l2/analyzes/bd_participation_ratio.yaml
./bin/1.3.l2/analyze/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --analysis-version "bd-pr;${ANALYSIS_VERSION}" \
    --epochs 300


# Spectral norm
ANALYSIS_CONF=configs/1.3.l2/analyzes/spectral_norm.yaml
./bin/1.3.l2/analyze/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --analysis-version "spectral-norm;${ANALYSIS_VERSION}" \
    --epochs 300


# l2 norm
ANALYSIS_CONF=configs/1.3.l2/analyzes/l2.yaml
./bin/1.3.l2/analyze/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --analysis-version "l2-norm;${ANALYSIS_VERSION}" \
    --epochs 300


./bin/1.3.l2/analyze/deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --execution-version ${EXECUTION_VERSION} --analysis-version ${ANALYSIS_VERSION}
