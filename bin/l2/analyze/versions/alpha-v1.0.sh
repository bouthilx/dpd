#!/usr/bin/env bash

###
# First execution of all analyses
###

FILENAME="$(basename "$0")"
ANALYSIS_VERSION="a-${FILENAME%.*}"
EXECUTION_VERSION="alpha-v1.1"
CONTAINER="1a4016b"


# # Simplified Fisher-Rao norm
# ANALYSIS_CONF=configs/1.3.l2/analyzes/fisher_rao_norm.yaml
# ./bin/1.3.l2/analyze/submit.sh $CONTAINER ${EXECUTION_VERSION} \
#     "fisher-rao-norm;${ANALYSIS_VERSION}" ${ANALYSIS_CONF} \
#     --models mlp1wb mlp2wb mlp5wb --epochs 300
#     
# 
# # KFAC Fisher-Rao norm
# ANALYSIS_CONF=configs/1.3.l2/analyzes/kfac_fisher_rao_norm.yaml
# ./bin/1.3.l2/analyze/submit.sh $CONTAINER ${EXECUTION_VERSION} \
#     "kfac-fisher-rao-norm;${ANALYSIS_VERSION}" ${ANALYSIS_CONF} \
#     --epochs 300
#     
# 
# # Participation ratio
# ANALYSIS_CONF=configs/1.3.l2/analyzes/participation_ratio.yaml
# ./bin/1.3.l2/analyze/submit.sh $CONTAINER ${EXECUTION_VERSION} \
#     "pr;${ANALYSIS_VERSION}" ${ANALYSIS_CONF} \
#     --models mlp1 mlp2 mlp1wb mlp2wb --epochs 300
#     
# 
# # Block Diagonal Participation ratio
# ANALYSIS_CONF=configs/1.3.l2/analyzes/bd_participation_ratio.yaml
# ./bin/1.3.l2/analyze/submit.sh $CONTAINER ${EXECUTION_VERSION} \
#     "bd-pr;${ANALYSIS_VERSION}" ${ANALYSIS_CONF} \
#     --epochs 300
# 
# 
# # Spectral norm
# ANALYSIS_CONF=configs/1.3.l2/analyzes/spectral_norm.yaml
# ./bin/1.3.l2/analyze/submit.sh $CONTAINER ${EXECUTION_VERSION} \
#     "spectral-norm;${ANALYSIS_VERSION}" ${ANALYSIS_CONF} \
#     --epochs 300
# 
# 
# # l2 norm
# ANALYSIS_CONF=configs/1.3.l2/analyzes/l2.yaml
# ./bin/1.3.l2/analyze/submit.sh $CONTAINER ${EXECUTION_VERSION} \
#     "l2-norm;${ANALYSIS_VERSION}" ${ANALYSIS_CONF} \
#     --epochs 300


./bin/1.3.l2/analyze/deploy.sh bouthilx/sgd-space-hub:$CONTAINER \
    --execution-version ${EXECUTION_VERSION} --analysis-version ${ANALYSIS_VERSION}
