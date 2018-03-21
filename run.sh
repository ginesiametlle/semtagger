#!/bin/bash
# this is a general setup script for this project


# train a tagger model with option --train, -t
PARAMS_TRAIN=0

# predict sem-tags for unlabeled data with option --predict, -p
PARAMS_PREDICT=0

# point to a file containing untagged sentence data with option --input, -i
PRED_INPUT=${DIR_DATA}/sample/sample_en.off

# point to a file containing tag predictions for the input file with option --output, -o
PRED_OUTPUT=${DIR_DATA}/sample/sample_en.sem


# set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands'
set -e
set -u
set -o pipefail
#set -x

# ensure script runs from the root directory
DIR_ROOT=${PWD}
if ! [ -x ${DIR_ROOT}/run.sh ]; then
    echo '[INFO] You must execture run.sh from the root directory'
    exit 1
fi

# load configuration options
. ${DIR_ROOT}/config.sh

# transform long options to short ones and parse them
for arg in "$@"; do
    shift
    case "$arg" in
        "--train") set -- "$@" "-t" ;;
        "--predict") set -- "$@" "-p" ;;
        "--input") set -- "$@" "-i" ;;
        "--output") set -- "$@" "-o" ;;
        *) set -- "$@" "$arg"
    esac
done

while getopts s:tpio option
do
    case "${option}"
    in
        t) PARAMS_TRAIN=1;;
        p) PARAMS_PREDICT=1;;
        i) PRED_INPUT=1;;
        o) PRED_OUTPUT=1;;
    esac
done


if [ ! ${PARAMS_TRAIN} -eq 0 ]; then
	  # DOWNLOAD AND PREPARE DATA
	  echo '[INFO] Preparing data...'
	  . ${DIR_DATA}/prepare_data.sh
	  echo '[INFO] Finished preparing data'

	  # SETUP REQUIRED TOOLS
	  echo '[INFO] Setting up required tools...'
	  . ${DIR_TOOLS}/prepare_tools.sh
	  echo '[INFO] Finished setting up tools'

	  # TRAIN A MODEL
	  echo "[INFO] Training ${MODEL_TYPE} models for semantic tagging..."
	  . ${DIR_MODELS}/semtagger_train.sh
	  echo "[INFO] A ${MODEL_TYPE} model was succesfully trained"
fi


if [ ! ${PARAMS_PREDICT} -eq 0 ]; then
	  # PREDICT USING A TRAINED MODEL
	  echo "[INFO] Predicting sem-tags using a ${MODEL_TYPE} model..."
	  . ${DIR_MODELS}/semtagger_predict.sh
    echo '[INFO] Finished tagging'
fi

