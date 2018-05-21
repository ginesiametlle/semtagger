#!/bin/bash
# this script trains a new semantic tagger on the input data


# employ a model for each target language
for idx in ${!PMB_LANGS[*]} ; do
    l=${PMB_LANGS[$idx]}
    LWEMB=${EMB_WORD_PRETRAINED[$idx]}
    LCEMB=${EMB_CHAR_PRETRAINED[$idx]}
    LWEMB_TRAINABLE=0
    LCEMB_TRAINABLE=0

    MODEL_GIVEN_PATH="${MODEL_GIVEN_DIR}/${l}/tagger.hdf5"
    MODEL_PATH_INFO="${MODEL_GIVEN_DIR}/${l}/tagger_params.pkl"

    # use default or pretrained word embeddings
    if [ ! -f ${LWEMB} ] || [ -z ${LWEMB} ] && [ ! ${EMB_USE_WORDS} -eq 0 ]; then
        LWEMB=${EMB_ROOT}/${l}/wemb_${l}.txt
        LWEMB_TRAINABLE=0
    fi

    # use default or pretrained character embeddings
    if [ ! -f ${LCEMB} ] || [ -z ${LCEMB} ] && [ ! ${EMB_USE_CHARS} -eq 0 ]; then
        LCEMB=${EMB_ROOT}/${l}/cemb_${l}.txt
        LCEMB_TRAINABLE=1
    fi

    # use an existing model if it exists
    if [ -f ${MODEL_GIVEN_PATH} ] && [ ${GET_MODEL} -eq 0 ]; then
        echo "[INFO] A matching trained model was found for '${l}'"
        echo "[INFO] Using the model in ${MODEL_GIVEN_PATH}"
    # train a new model
    else
        echo "[INFO] Training a new model for language '${l}'..."
        echo "[INFO] The new model will be stored in ${MODEL_GIVEN_PATH}"
        rm -f ${MODEL_GIVEN_PATH}
        mkdir -p $(dirname ${MODEL_GIVEN_PATH})

        rm -f ${PMB_EXTDIR}/${l}/wsents_${l}.sem
        rm -f ${PMB_EXTDIR}/${l}/csents_${l}.txt

        python3 ${DIR_MODELS}/semtagger_fit.py ${DIR_ROOT} \
                --raw_pmb_data ${PMB_EXTDIR}/${l}/pmb_${l}.sem \
                --raw_extra_data ${PMB_EXTDIR}/${l}/extra_${l}.sem \
                --output_words ${PMB_EXTDIR}/${l}/wsents_${l}.sem \
                --output_chars ${PMB_EXTDIR}/${l}/csents_${l}.txt \
                --word_embeddings ${LWEMB} \
                --char_embeddings ${LCEMB} \
                --word_embeddings_trainable ${LWEMB_TRAINABLE} \
                --char_embeddings_trainable ${LCEMB_TRAINABLE} \
                --use_words ${EMB_USE_WORDS} \
                --use_chars ${EMB_USE_CHARS} \
                --output_model ${MODEL_GIVEN_PATH} \
                --output_model_info ${MODEL_PATH_INFO} \
                --lang ${l} \
                --model ${MODEL_TYPE} \
                --epochs ${MODEL_EPOCHS} \
                --model_size ${MODEL_SIZE} \
                --num_layers ${MODEL_LAYERS} \
                --noise_sigma ${MODEL_SIGMA} \
                --hidden_activation ${MODEL_ACTIVATION_HIDDEN} \
                --output_activation ${MODEL_ACTIVATION_OUTPUT} \
                --loss ${MODEL_LOSS} \
                --optimizer ${MODEL_OPTIMIZER} \
                --dropout ${MODEL_DROPOUT} \
                --batch_size ${MODEL_BATCH_SIZE} \
                --batch_normalization ${MODEL_BATCH_NORMALIZATION} \
                --verbose ${MODEL_VERBOSE} \
                --test_size ${RUN_TEST_SIZE} \
                --dev_size ${RUN_DEV_SIZE} \
                --grid_search ${RUN_GRID_SEARCH} \
                --sent_len_perc ${RUN_SENT_LEN} \
                --word_len_perc ${RUN_WORD_LEN} \
                --multi_word ${RUN_MWE} \
                --resnet_depth ${RUN_RESNET_DEPTH}
    fi
done

