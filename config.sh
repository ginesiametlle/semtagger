#!/bin/bash
# this script defines configuration options

######################
## BASE DIRECTORIES ##
######################

# data directory (string)
DIR_DATA=${DIR_ROOT}/data

# tools directory (string)
DIR_TOOLS=${DIR_ROOT}/tools

# models directory (string)
DIR_MODELS=${DIR_ROOT}/models

# utils directory (string)
DIR_UTILS=${DIR_ROOT}/utils

#############
## UPDATES ##
#############

# downloads and processes data from the PMB overwriting existing data (boolean, default: 0)
GET_PMB=0

# downloads and processes word embeddings overwriting existing ones (boolean, default: 0)
GET_EMBS=0

# downloads and installs required tools overwriting previous versions (boolean, default: 0)
GET_TOOLS=0

# trains a semantic tagger model overwritting existing models (boolean, default: 0)
GET_MODEL=0

###########################
## PARALLEL MEANING BANK ##
###########################

# version of the PMB to use (string)
PMB_VER="1.0.0"

# root directory where to store the PMB (string)
PMB_ROOT=${DIR_DATA}/pmb-${PMB_VER}

# languages of the PMB (iso 639-1) for which to extract tagged sentences (array)
# allowed values: "en", "de", "it", "nl"
PMB_LANGS=("en")

######################
## GLOVE EMBEDDINGS ##
######################

# root directory where to store embeddings (string)
GLOVE_ROOT=${DIR_DATA}/embeddings

# pretrained version of the embeddings to use (string)
# allowed values: "glove.6B", "glove.42B.300d", "glove.840B.300d"
GLOVE_MODEL="glove.6B"

######################
## MODEL PARAMETERS ##
######################

# type of neural model to use (string)
# allowed values: "lstm", "lstm-crf", "bi-lstm", "bi-lstm-crf"
MODEL_TYPE="lstm"

# directory where to store the trained models (string)
MODEL_ROOT=${DIR_MODELS}/${MODEL_TYPE}

# training iterations (int, default: 30)
MODEL_ITERS=10

# hidden units of the neural model (int, default: 100)
MODEL_SIZE_HIDDEN=10

# noise parameter sigma (float, default: 0.2)
MODEL_SIGMA=0.2

# activation function on hidden layers (string)
# allowed values: "tanh"
MODEL_ACTIVATION_HIDDEN="tanh"

# activation function on the output layer (string)
# allowed values: "sigmoid"
MODEL_ACTIVATION_OUTPUT="sigmoid"

# optimizer (string)
# allowed values: "sgd"
MODEL_OPTIMIZER="sgd"

# loss function (string)
# allowed values: "mse"
MODEL_LOSS="mse"

# learning rate (float, default: 0.1)
MODEL_LEARNING_RATE=0.1

# dropout rate on each layer (float, default: 0.2)
MODEL_DROPOUT=0.2

# batch size (int, default: 32)
MODEL_BATCH_SIZE=32

##########################
## TRAINING AND TESTING ##
##########################

# proportion of tagged sentences to use for training (float, default: 0.80)
RUN_TRAIN_RATIO=0.80

# estimate hyperparameters on cross-validation or use fixed values (boolean, default: false)
RUN_CROSS_VAL=0

#################
## PREDICTIONS ##
#################

# file containing untagged sentence data
PRED_INPUT=${DIR_DATA}/sample/toy.off

# file containing the tag predictions for the input file
PRED_OUTPUT=${DIR_DATA}/sample/toy.sem

