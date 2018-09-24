#!/usr/bin/env bash

###
# First execution of all analyses
###

FILENAME="$(basename "$0")"
ANALYSIS_VERSION="a-${FILENAME%.*}"
EXECUTION_VERSION="beta-v1.0"
CONTAINER="952efb7"


# Simplified Fisher-Rao norm
ANALYSIS_CONF=configs/1.synthetic/analyzes/fisher_rao_norm.yaml
./bin/1.synthetic/analyze/submit.sh $CONTAINER ${EXECUTION_VERSION} ${ANALYSIS_VERSION} ${ANALYSIS_CONF} \
    --models mlp1wb mlp2wb mlp5wb --epochs 300
    

# KFAC Fisher-Rao norm
ANALYSIS_CONF=configs/1.synthetic/analyzes/kfac_fisher_rao_norm.yaml
./bin/1.synthetic/analyze/submit.sh $CONTAINER ${EXECUTION_VERSION} ${ANALYSIS_VERSION} ${ANALYSIS_CONF} \
    --epochs 300
    

# Participation ratio
ANALYSIS_CONF=configs/1.synthetic/analyzes/participation_ratio.yaml
./bin/1.synthetic/analyze/submit.sh $CONTAINER ${EXECUTION_VERSION} ${ANALYSIS_VERSION} ${ANALYSIS_CONF} \
    --models mlp1 mlp2 mlp1wb mlp2wb --epochs 300
    

# Block Diagonal Participation ratio
ANALYSIS_CONF=configs/1.synthetic/analyzes/bd_participation_ratio.yaml
./bin/1.synthetic/analyze/submit.sh $CONTAINER ${EXECUTION_VERSION} ${ANALYSIS_VERSION} ${ANALYSIS_CONF} \
    --epochs 300


# # Spectral norm
# ANALYSIS_CONF=configs/1.synthetic/analyzes/spectral_norm.yaml
# ./bin/1.synthetic/analyze/submit.sh $CONTAINER ${EXECUTION_VERSION} ${ANALYSIS_VERSION} ${ANALYSIS_CONF} \
#     --epochs 300
    

# l2 norm
ANALYSIS_CONF=configs/1.synthetic/analyzes/l2.yaml
./bin/1.synthetic/analyze/submit.sh $CONTAINER ${EXECUTION_VERSION} ${ANALYSIS_VERSION} ${ANALYSIS_CONF} \
    --epochs 300
