#!/bin/bash
# this script trains a model and evaluates on all the available data subsets for a given language

#SBATCH --time=20:00:00
#SBATCH --mem=20GB

# root directory where this script is located
DIR_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# target language code
# allowed values: "en", "de", "it", "nl"
EVAL_LANG="en"

# load configuration options
. ${DIR_ROOT}/config.sh

# train a new model
. ${DIR_ROOT}/run.sh --train

# predict and evaluate on gold/train
if [ -f ${PMB_EXTDIR}/${EVAL_LANG}/gold/train.off ] && [ -f ${PMB_EXTDIR}/${EVAL_LANG}/gold/train.gold ]; then
	OPTIND=1
	. ${DIR_ROOT}/run.sh --predict --input ${PMB_EXTDIR}/${EVAL_LANG}/gold/train.off --output ${PMB_EXTDIR}/${EVAL_LANG}/gold/train.sem
	python3 ${DIR_ROOT}/utils/compare_tags.py ${PMB_EXTDIR}/${EVAL_LANG}/gold/train.sem ${PMB_EXTDIR}/${EVAL_LANG}/gold/train.gold
fi

# predict and evaluate on gold/test
if [ -f ${PMB_EXTDIR}/${EVAL_LANG}/gold/test.off ] && [ -f ${PMB_EXTDIR}/${EVAL_LANG}/gold/test.gold ]; then
	OPTIND=1
	. ${DIR_ROOT}/run.sh --predict --input ${PMB_EXTDIR}/${EVAL_LANG}/gold/test.off --output ${PMB_EXTDIR}/${EVAL_LANG}/gold/test.sem
	python3 ${DIR_ROOT}/utils/compare_tags.py ${PMB_EXTDIR}/${EVAL_LANG}/gold/test.sem ${PMB_EXTDIR}/${EVAL_LANG}/gold/test.gold
fi

# predict and evaluate on silver/train
if [ -f ${PMB_EXTDIR}/${EVAL_LANG}/silver/train.off ] && [ -f ${PMB_EXTDIR}/${EVAL_LANG}/silver/train.gold ]; then
	OPTIND=1
	. ${DIR_ROOT}/run.sh --predict --input ${PMB_EXTDIR}/${EVAL_LANG}/silver/train.off --output ${PMB_EXTDIR}/${EVAL_LANG}/silver/train.sem
	python3 ${DIR_ROOT}/utils/compare_tags.py ${PMB_EXTDIR}/${EVAL_LANG}/silver/train.sem ${PMB_EXTDIR}/${EVAL_LANG}/silver/train.gold
fi

# predict sem-tags for silver/test
if [ -f ${PMB_EXTDIR}/${EVAL_LANG}/silver/test.off ] && [ -f ${PMB_EXTDIR}/${EVAL_LANG}/silver/test.gold ]; then
	OPTIND=1
	. ${DIR_ROOT}/run.sh --predict --input ${PMB_EXTDIR}/${EVAL_LANG}/silver/test.off --output ${PMB_EXTDIR}/${EVAL_LANG}/silver/test.sem
	python3 ${DIR_ROOT}/utils/compare_tags.py ${PMB_EXTDIR}/${EVAL_LANG}/silver/test.sem ${PMB_EXTDIR}/${EVAL_LANG}/silver/test.gold
fi
