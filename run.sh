#!/bin/bash
# this is a general setup script for this project


# ensure script runs from the root directory
DIR_ROOT=${PWD}
if ! [ -x ${DIR_ROOT}/run.sh ]; then
    echo '[INFO] You must execture run.sh from the root directory'
    exit 1
fi

# load configuration options
. ${DIR_ROOT}/config.sh

# train a tagger model with option --train, -t
PARAMS_TRAIN=0

# predict sem-tags for unlabeled data with option --predict, -p
PARAMS_PREDICT=0

# point to a file containing untagged sentence data with option --input, -i
PRED_INPUT=${DIR_DATA}/sample/qa_en.off

# point to a file containing tag predictions for the input file with option --output, -o
PRED_OUTPUT=${DIR_DATA}/sample/qa_en.sem

# point to a file containing the model to store/load with option --model, -m
MODEL_GIVEN_PATH=${MODEL_ROOT}/${MODEL_TYPE}-${MODEL_SIZE}-${MODEL_LAYERS}-${MODEL_ACTIVATION_OUTPUT}-${l}.model

# set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands'
set -e
set -u
set -o pipefail
#set -x

# transform long options to short ones and parse them
for arg in "$@"; do
    shift
    case "$arg" in
        "--train") set -- "$@" "-t" ;;
        "--predict") set -- "$@" "-p" ;;
        "--input") set -- "$@" "-i" ;;
        "--output") set -- "$@" "-o" ;;
        "--model") set -- "$@" "-m" ;;
        *) set -- "$@" "$arg"
    esac
done

while getopts s:tpi:o:m: option
do
    case "${option}"
    in
        t) PARAMS_TRAIN=1;;
        p) PARAMS_PREDICT=1;;
        i) PRED_INPUT=${OPTARG};;
        o) PRED_OUTPUT=${OPTARG};;
        m) MODEL_GIVEN_PATH=${OPTARG};;
    esac
done

# check for correctness of the configuration file
n_pmb_langs=${#PMB_LANGS[@]}
n_pretrained=${#EMB_PRETRAINED[@]}
n_extra_files=${#PMB_EXTRA_SRC[@]}
n_extra_langs=${#PMB_EXTRA_LANGS[@]}

if [ ${n_pmb_langs} -ne ${n_pretrained} ]; then
    echo "[ERROR] The specified numbers of PMB languages and their corresponding pretrained embedding files do not match (please fix config.sh)"
    exit
fi

if [ ${n_extra_files} -ne ${n_extra_langs} ] && [ ! ${PMB_EXTRA_DATA} -eq 0 ]; then
    echo "[ERROR] The specified numbers of additional data files and their corresponding languages do not match (please fix config.sh)"
    exit
fi


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

