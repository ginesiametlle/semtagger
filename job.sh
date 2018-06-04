#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH --mem=20GB

TAGGER_HOME="/home/joan/semtagger"
PMB_HOME="${TAGGER_HOME}/data/pmb"

# train a new model
. ${TAGGER_HOME}/run.sh --train

# predict sem-tags for gold/train
OPTIND=1
. ${TAGGER_HOME}/run.sh --predict --input ${PMB_HOME}/en/gold/gold_train.off --output ${PMB_HOME}/en/gold/gold_train.sem

# predict sem-tags for gold/test
OPTIND=1
. ${TAGGER_HOME}/run.sh --predict --input ${PMB_HOME}/en/gold/gold_test.off --output ${PMB_HOME}/en/gold/gold_test.sem

# predict sem-tags for silver/train
OPTIND=1
. ${TAGGER_HOME}/run.sh --predict --input ${PMB_HOME}/en/silver/silver_train.off --output ${PMB_HOME}/en/silver/silver_train.sem

# predict sem-tags for silver/test
OPTIND=1
. ${TAGGER_HOME}/run.sh --predict --input ${PMB_HOME}/en/silver/silver_test.off --output ${PMB_HOME}/en/silver/silver_test.sem

# predict sem-tags for WebQuestions
OPTIND=1
. ${TAGGER_HOME}/run.sh --predict --input ${TAGGER_HOME}/qa/sample/questions.off --output ${TAGGER_HOME}/qa/sample/questions.sem

# evaluate accuracy on gold/train
python3 ${TAGGER_HOME}/utils/compare_tags.py ${PMB_HOME}/en/gold/gold_train.sem ${PMB_HOME}/en/gold/train/gold_train.gold

# evaluate accuracy on gold/test
python3 ${TAGGER_HOME}/utils/compare_tags.py ${PMB_HOME}/en/gold/gold_test.sem ${PMB_HOME}/en/gold/test/gold_test.gold

# evaluate accuracy on silver/train
python3 ${TAGGER_HOME}/utils/compare_tags.py ${PMB_HOME}/en/silver/silver_train.sem ${PMB_HOME}/en/silver/train/silver_train.gold

# evaluate accuracy on silver/test
python3 ${TAGGER_HOME}/utils/compare_tags.py ${PMB_HOME}/en/silver/silver_test.sem ${PMB_HOME}/en/silver/test/silver_test.gold

# evaluate accuracy WebQuestions
python3 ${TAGGER_HOME}/utils/compare_tags.py ${TAGGER_HOME}/qa/sample/questions.sem ${TAGGER_HOME}/qa/sample/questions.gold

