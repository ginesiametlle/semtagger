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

# processes the additional data again overwriting existing data (boolean, default: 0)
GET_EXTRA=0

# downloads and processes word embeddings overwriting existing ones (boolean, default: 0)
GET_EMBS=0

# downloads and installs required tools overwriting previous versions (boolean, default: 0)
GET_TOOLS=0

# trains a semantic tagger model overwritting existing models (boolean, default: 0)
GET_MODEL=0

###########################
## PARALLEL MEANING BANK ##
###########################

# use semantically tagged data from the PMB (boolean, default: 1)
PMB_MAIN_DATA=1

# version of the PMB Universal Semantic Tags release to use (string)
# currently available versions are "0.1.0"
PMB_VER="0.1.0"

# root directory where to store the PMB (string)
PMB_ROOT=${DIR_DATA}/pmb/sem-${PMB_VER}

# directory where to store data extracted from the PMB (string)
PMB_EXTDIR=${DIR_DATA}/pmb

# codes of PMB languages for which to extract tagged sentences (array)
# allowed values: "en", "de", "it", "nl"
PMB_LANGS=("en")

# proportion of PMB tagged sentences to include in the test set (float, default: 0.10)
# the remaining sentences are included in the training set
PMB_TEST_SIZE=0.10

# use additional semantically tagged data (boolean, default: 0)
# set this option to 0 if you do not have access to additional data
PMB_EXTRA_DATA=0

# directories with additional semantically tagged data (array)
# each directory listed is assumed to contain a number of files
# each file is assumed to contain [TAG]\t[WORD] lines
# each file is assumed to contain empty lines denoting the end of a sentence
PMB_EXTRA_SRC=()

# languages corresponding to the data of each directory with extra data (array)
# allowed values: "en", "de", "it", "nl"
PMB_EXTRA_LANGS=()

################
## EMBEDDINGS ##
################

# whether or not to use word embeddings (boolean, default: 1)
EMB_USE_WORDS=1

# whether or not to use character embeddings (boolean, default: 1)
EMB_USE_CHARS=1

# pretrained word embeddings for each one of the PMB languages (array)
# the files listed are assumed to be in the same order as PMB_LANGS
# default embeddings are used when a given string is empty or does not match a file
EMB_WORD_PRETRAINED=("")

# pretrained character embeddings for each one of the PMB languages (array)
# the files listed are assumed to be in the same order as PMB_LANGS
# default embeddings are used when a given string is empty or does not match a file
EMB_CHAR_PRETRAINED=("")

# root directory where to store word embeddings for English (string)
# GloVe embeddings are only used if no pretrained embeddings are given
# languages other than English use Polyglot embeddings instead
EMB_ROOT=${DIR_DATA}/embeddings

# version of the GloVe word embeddings to use for English as default (string)
# allowed values: "glove.6B.{50/100/200/300}d", "glove.42B.300d", "glove.840B.300d"
EMB_GLOVE_MODEL="glove.840B.300d"

##########################
## TRAINING AND TESTING ##
##########################

# proportion of sentences in the training set to use for testing purposes (float, default: 0.00)
# these sentences are deducted from the training set and are evaluated after training
RUN_TEST_SIZE=0.00

# proportion of sentences in the training set to use for development purposes (float, default: 0.00)
# these sentences are deducted from the training set and are evaluated after each training epoch
RUN_DEV_SIZE=0.00

# run grid-search for hyperparameter optimization (boolean, default: 0)
# grid-search is time-consuming and can change the hyperparameters defined in this file
# the hyperparameters here defined are shared among models for all languages otherwise
RUN_GRID_SEARCH=0

# maximum sentence length allowed, as a percentile on the sentence length distribution (float, default: 0.95)
# the number of words in a sentence for creating word-based features is computed based on this number
RUN_SENT_LEN=0.95

# maximum word length allowed, as a percentile on the word length distribution (float, default: 0.98)
# the number of characters in a word for creating character-based features is computed based on this number
RUN_WORD_LEN=0.98

# handle multi-word expressions (boolean, default: 1)
RUN_MWE=1

# depth of the residual network applied on character embedding features (int, default: 6)
# the residual network helps turn character embeddings into word-like representations
RUN_RESNET_DEPTH=6

#####################
# MODEL PARAMETERS ##
#####################

# type of neural model to use (string)
# allowed values: "lstm", "blstm", "gru", "bgru"
MODEL_TYPE="bgru"

# directory where to store the trained model (string)
MODEL_ROOT=${DIR_MODELS}/bin

# training epochs (int, default: 10)
MODEL_EPOCHS=10

# units in the first layer of the neural model (int, default: 300)
MODEL_SIZE=300

# number of recurrent layers of the neural model (int, default: 1)
MODEL_LAYERS=1

# standard deviation for the noise normal distribution (float, default: 0.0)
MODEL_SIGMA=0.0

# activation function on hidden layers (string)
# allowed values: "sigmoid", "tanh", "relu"
MODEL_ACTIVATION_HIDDEN="relu"

# type of output layer (string)
# allowed values: "softmax", "crf"
MODEL_ACTIVATION_OUTPUT="crf"

# loss function (string)
# allowed values: "mean_squared_error", "mean_absolute_error", "categorical_hinge", "categorical_cross_entropy"
MODEL_LOSS="categorical_cross_entropy"

# optimizer (string)
# allowed values: "sgd", "rmsprop", "adam", "adamax", "nadam"
MODEL_OPTIMIZER="adamax"

# dropout rate on each layer (float, default: 0.16)
MODEL_DROPOUT=0.16

# batch size (int, default: 150)
MODEL_BATCH_SIZE=150

# use batch normalization (boolean, default: 1)
MODEL_BATCH_NORMALIZATION=1

# keras verbosity mode (int, default: 1)
MODEL_VERBOSE=1

