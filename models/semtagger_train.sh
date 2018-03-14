#!/bin/bash
# this script trains a new semantic tagger on the input data

# this script has to call model_fit.py on the right input data, train ratio, model, etc
# semtagger_fit will need a lot parameters

# train a model for each target language
for l in ${PMB_LANGS[@]} ; do
    python3 ${DIR_MODELS}/semtagger_fit.py \
            --data ${PMB_EXTDIR}/${l}/sents_${l}.sem \
            --embeddings ${GLOVE_ROOT}/${GLOVE_MODEL}.txt \
            --output ${DIR_MODELS}/bin/${MODEL_TYPE}-${MODEL_SIZE_HIDDEN}.model \
            --lang ${l} \
            --test_size ${RUN_TEST_SIZE} \
            --max_sent_len ${RUN_MAX_LEN}
done
