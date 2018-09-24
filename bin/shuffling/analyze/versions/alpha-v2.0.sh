#!/usr/bin/env bash

###
# Second execution of all analyses.
# Run on epochs 1,50,100,150,200,250,300
# Fix KFAC computation on loss
# Add shuffling product norm
# Use 1000 batches for PR metrics
###


FILENAME="$(basename "$0")"
ANALYSIS_VERSION="a-${FILENAME%.*}"
EXECUTION_VERSION="alpha-v2.0"
CONTAINER="bouthilx/sgd-space-hub:495d511"


VALUES="'epochs=list(range(6))+[10, 15, 20, 25, 50, 75, 100, 150, 200, 250, 300]'"


# Simplified Fisher-Rao norm
ANALYSIS_CONF=configs/shuffling/analyzes/fisher_rao_norm.yaml
./bin/register.sh $CONTAINER analyses \
    --experiment shuffling \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --version "${ANALYSIS_VERSION}" \
    --tags "fisher-rao-norm" \
    --models mlp1wb mlp2wb mlp5wb \
    --values "$VALUES"


# KFAC Fisher-Rao norm
ANALYSIS_CONF=configs/shuffling/analyzes/kfac_fisher_rao_norm.yaml
./bin/register.sh $CONTAINER analyses \
    --experiment shuffling \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --version "${ANALYSIS_VERSION}" \
    --tags "kfac-fisher-rao-norm" \
    --values "$VALUES"


# Participation ratio
ANALYSIS_CONF=configs/shuffling/analyzes/participation_ratio.yaml
./bin/register.sh $CONTAINER analyses \
    --experiment shuffling \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --version "${ANALYSIS_VERSION}" \
    --tags "pr" \
    --models mlp1 mlp2 mlp1wb mlp2wb \
    --values "$VALUES"


# Block Diagonal Participation ratio
ANALYSIS_CONF=configs/shuffling/analyzes/bd_participation_ratio.yaml
./bin/register.sh $CONTAINER analyses \
    --experiment shuffling \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --version "${ANALYSIS_VERSION}" \
    --tags "bd-pr" \
    --models mlp1 mlp2 mlp1wb mlp2wb \
    --values "$VALUES"


# Spectral norm
ANALYSIS_CONF=configs/shuffling/analyzes/spectral_norm.yaml
./bin/register.sh $CONTAINER analyses \
    --experiment shuffling \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --version "${ANALYSIS_VERSION}" \
    --tags "spectral-norm" \
    --values "$VALUES"


# shuffling norm
ANALYSIS_CONF=configs/shuffling/analyzes/l2.yaml
./bin/register.sh $CONTAINER analyses \
    --experiment shuffling \
    --config ${ANALYSIS_CONF} \
    --execution-version ${EXECUTION_VERSION} \
    --version "${ANALYSIS_VERSION}" \
    --tags "l2-norm" \
    --values "$VALUES"


# Schedule jobs with sbatch
./bin/schedule.sh ${CONTAINER} analyses \
    --experiment shuffling \
    --version "${ANALYSIS_VERSION}"
