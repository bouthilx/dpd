#!/usr/bin/env bash

###
# Second execution of all analyses.
# Run on epochs 1,50,100,150,200,250,300
# Fix KFAC computation on loss
# Add l2 product norm
###


FILENAME="$(basename "$0")"
ANALYSIS_VERSION="a-${FILENAME%.*}"
EXECUTION_VERSION="alpha-v1.1"
CONTAINER="495d511"


# Simplified Fisher-Rao norm
ANALYSIS_CONF=configs/1.3.l2/analyzes/fisher_rao_norm.yaml
./bin/1.3.l2/analyze/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --analysis-version "fisher-rao-norm;analysis;${ANALYSIS_VERSION}" \
    --models mlp1wb mlp2wb mlp5wb --epochs 1,50,100,150,200,250,300


# KFAC Fisher-Rao norm
ANALYSIS_CONF=configs/1.3.l2/analyzes/kfac_fisher_rao_norm.yaml
./bin/1.3.l2/analyze/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --analysis-version "kfac-fisher-rao-norm;analysis;${ANALYSIS_VERSION}" \
    --epochs 1,50,100,150,200,250,300


# Participation ratio
ANALYSIS_CONF=configs/1.3.l2/analyzes/participation_ratio.yaml
./bin/1.3.l2/analyze/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --analysis-version "pr;analysis;${ANALYSIS_VERSION}" \
    --models mlp1 mlp2 mlp1wb mlp2wb --epochs 1,50,100,150,200,250,300


# Block Diagonal Participation ratio
ANALYSIS_CONF=configs/1.3.l2/analyzes/bd_participation_ratio.yaml
./bin/1.3.l2/analyze/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --analysis-version "bd-pr;analysis;${ANALYSIS_VERSION}" \
    --epochs 1,50,100,150,200,250,300


# # Spectral norm
# ANALYSIS_CONF=configs/1.3.l2/analyzes/spectral_norm.yaml
# ./bin/1.3.l2/analyze/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
#     --config ${ANALYSIS_CONF} \
#     --execution-version ${EXECUTION_VERSION} \
#     --analysis-version "spectral-norm;analysis;${ANALYSIS_VERSION}" \
#     --epochs 1,50,100,150,200,250,300


# l2 norm
ANALYSIS_CONF=configs/1.3.l2/analyzes/l2.yaml
./bin/1.3.l2/analyze/pre_deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --analysis-version "l2-norm;analysis;${ANALYSIS_VERSION}" \
    --epochs 1,50,100,150,200,250,300


./bin/1.3.l2/analyze/deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --analysis-version "analysis;${ANALYSIS_VERSION}"
