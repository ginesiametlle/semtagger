#!/bin/bash
# this is a general setup script for this project


# root directory where this script is located
DIR_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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

# point to a directory containing the model to store/load with option --model, -m
MODEL_GIVEN_DIR=${MODEL_ROOT}/${MODEL_TYPE}-${MODEL_SIZE}-${MODEL_LAYERS}-${MODEL_ACTIVATION_OUTPUT}
if [ ! ${EMB_USE_WORDS} -eq 0 ]; then
	  MODEL_GIVEN_DIR="${MODEL_GIVEN_DIR}-words"
fi
if [ ! ${EMB_USE_CHARS} -eq 0 ]; then
	  MODEL_GIVEN_DIR="${MODEL_GIVEN_DIR}-chars"
fi

# space used for aligning messages to the user
HSPACE='   '


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
        m) MODEL_GIVEN_DIR=${OPTARG};;
    esac
done


# check for correctness of the configuration file
n_pmb_langs=${#PMB_LANGS[@]}
n_wpretrained=${#EMB_WORD_PRETRAINED[@]}
n_cpretrained=${#EMB_CHAR_PRETRAINED[@]}
n_extra_files=${#PMB_EXTRA_SRC[@]}
n_extra_langs=${#PMB_EXTRA_LANGS[@]}

if [ ${n_pmb_langs} -ne ${n_wpretrained} ]; then
    echo "[ERROR] The specified number of PMB languages and the provided pretrained word embedding files do not match"
    exit
fi

if [ ${n_pmb_langs} -ne ${n_cpretrained} ]; then
    echo "[ERROR] The specified number of PMB languages and the provided pretrained character embedding files do not match"
    exit
fi

if [ ${n_extra_files} -ne ${n_extra_langs} ] && [ ! ${PMB_EXTRA_DATA} -eq 0 ]; then
    echo "[ERROR] The specified number of additional data files and their corresponding languages do not match"
    exit
fi


if [ ! ${PARAMS_TRAIN} -eq 0 ]; then
    # SETUP REQUIRED TOOLS
    echo '[INFO] Setting up required tools...'
    . ${DIR_TOOLS}/prepare_tools.sh
    echo '[INFO] Finished setting up tools'

    # DOWNLOAD AND PREPARE DATA
    echo '[INFO] Preparing data...'
    . ${DIR_DATA}/prepare_data.sh
    echo '[INFO] Finished preparing data'

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

