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
GET_MODEL=1

###########################
## PARALLEL MEANING BANK ##
###########################

# version of the PMB to use (string)
PMB_VER="1.0.0"

# root directory where to store the PMB (string)
PMB_ROOT=${DIR_DATA}/pmb/pmb-${PMB_VER}

# directory where to store data extracted from the PMB (string)
PMB_EXTDIR=${DIR_DATA}/pmb

# languages of the PMB (iso 639-1) for which to extract tagged sentences (array)
# allowed values: "en", "de", "it", "nl"
PMB_LANGS=("en")

######################
## GLOVE EMBEDDINGS ##
######################

# root directory where to store embeddings (string)
GLOVE_ROOT=${DIR_DATA}/embeddings

# pretrained version of the embeddings to use (string)
# allowed values: "glove.6B.{50/100/200/300}d", "glove.42B.300d", "glove.840B.300d"
GLOVE_MODEL="glove.6B.50d"

######################
## MODEL PARAMETERS ##
######################

# type of neural model to use (string)
# allowed values: "rnn", "lstm", "blstm", "gru", "bgru"
MODEL_TYPE="blstm"

# directory where to store the trained model (string)
MODEL_ROOT=${DIR_MODELS}/bin

# training epochs (int, default: 10)
MODEL_EPOCHS=10

# units in the first layer of the neural model (int, default: 50)
MODEL_SIZE=50

# number of recurrent layers of the neural model
# note that the number of hidden units is halved on each layer (int, default: 2)
MODEL_LAYERS=2

# noise parameter sigma (float, default: 0.1)
MODEL_SIGMA=0.1

# activation function on hidden layers (string)
# allowed values: "tanh", "relu"
MODEL_ACTIVATION_HIDDEN="tanh"

# activation function on the output layer (string)
# allowed values: "sigmoid", "crf"
MODEL_ACTIVATION_OUTPUT="sigmoid"

# loss function (string)
# allowed values: "mse"
MODEL_LOSS="mse"

# optimizer (string)
# allowed values: "sgd", "adam"
MODEL_OPTIMIZER="sgd"

# learning rate (float, default: 0.1)
MODEL_LEARNING_RATE=0.1

# dropout rate on each layer (float, default: 0.1)
MODEL_DROPOUT=0.1

# batch size (int, default: 32)
MODEL_BATCH_SIZE=32

# use batch normalization (boolean, default: 1)
MODEL_BATCH_NORMALIZATION=1

# keras verbose (boolean, default: False)
MODEL_VERBOSE=0

##########################
## TRAINING AND TESTING ##
##########################

# proportion of tagged sentences to use for training (float, default: 0.80)
RUN_TEST_SIZE=0.20

# run cross-validation (boolean, default: 0)
RUN_CROSS_VAL=0

# maximum sequence length allowed as a percentile (float, default: 0.9)
RUN_LEN_PERC=0.85

# handle multi-word expressions (boolean, default: 1)
RUN_MWE=1

#################
## OTHER TOOLS ##
#################

# root directory where to store the Elephant tokenizer
ELEPHANT_DIR="${DIR_TOOLS}/elephant"

