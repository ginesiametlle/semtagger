#!/bin/bash
# this script downloads and installs external tools


# download and prepare the Elephant tokenizer
echo '[INFO] Preparing the Elephant tokenizer...'
if [ ! -d "${ELEPHANT_DIR}" ] || [ ! ${GET_TOOLS} -eq 0 ]; then
    rm -rf ${ELEPHANT_DIR}
    mkdir -p ${ELEPHANT_DIR}
    pushd ${ELEPHANT_DIR} > /dev/null
    wget -q --show-progress "https://github.com/hslh/elephant/archive/master.zip"
    unzip -qq "master.zip"
    rm -f "master.zip"
    mv elephant-master/* .
    rm -rf elephant-master
    make > /dev/null
    popd > /dev/null
fi


# download and prepare the lm_1b model
echo '[INFO] Preparing the lm_1b model...'
if [ ! -d "${LM1B_DIR}" ] || [ ! ${GET_TOOLS} -eq 0 ]; then
    rm -rf ${LM1B_DIR}
    mkdir -p ${LM1B_DIR}
    pushd ${LM1B_DIR} > /dev/null
    # lm_1b directory
    mkdir -p ${LM1B_DIR}/lm_1b
    wget -q --show-progress "https://github.com/colinmorris/lm1b-notebook/archive/master.zip"
    unzip -qq "master.zip"
    rm -f "master.zip"
    mv lm1b-notebook-master/lm_1b/* ${LM1B_DIR}/lm_1b
    rm -rf lm1b-notebook-master
    awk 'NR==28{$0="WORD_DUMP_LIMIT = 20000"}1;' ${LM1B_DIR}/lm_1b/lm_1b_eval.py > tmp && mv tmp lm_1b_eval.py;
    # data directory
    mkdir -p ${LM1B_DIR}/data
    pushd ${LM1B_DIR}/data > /dev/null
    wget -q --show-progress "download.tensorflow.org/models/LM_LSTM_CNN/graph-2016-09-10.pbtxt"
    wget -q --show-progress "download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-base"
    wget -q --show-progress "download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-char-embedding"
    wget -q --show-progress "download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-lstm"
    wget -q --show-progress "download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax0"
    wget -q --show-progress "download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax1"
    wget -q --show-progress "download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax2"
    wget -q --show-progress "download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax3"
    wget -q --show-progress "download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax4"
    wget -q --show-progress "download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax5"
    wget -q --show-progress "download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax6"
    wget -q --show-progress "download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax7"
    wget -q --show-progress "download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax8"
    wget -q --show-progress "download.tensorflow.org/models/LM_LSTM_CNN/vocab-2016-09-10.txt"
    wget -q --show-progress "download.tensorflow.org/models/LM_LSTM_CNN/test/news.en.heldout-00000-of-00050"
    shuf -n 20000 ${LM1B_DIR}/data/vocab-2016-09-10.txt > ${LM1B_DIR}/data/vocab-small.txt
    popd > /dev/null
    # output directory
    mkdir -p ${LM1B_DIR}/output/filters
    # WORKSPACE file
    touch WORKSPACE
    # build the codes
    bazel build -c opt lm_1b/...
    popd > /dev/null
fi

