#!/bin/bash
# this script trains a new semantic tagger on the input data


# employ a model for each target language
for l in ${PMB_LANGS[@]} ; do

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

        # determine the location of the embeddings given the language
        if [ ${l} == "en" ]; then
            FIT_WEMB=${EMB_ROOT}/${l}/${EMB_GLOVE_MODEL}.txt
        else
            FIT_WEMB=${EMB_ROOT}/${l}/polyglot_${l}.txt
        fi

        rm -f ${PMB_EXTDIR}/${l}/sents_${l}.sem
        rm -f ${PMB_EXTDIR}/${l}/sents_${l}_chars.txt

        python3 ${DIR_MODELS}/semtagger_fit.py ${DIR_ROOT} \
                --raw_pmb_data ${PMB_EXTDIR}/${l}/pmb_${l}.sem \
                --raw_extra_data ${PMB_EXTDIR}/${l}/extra_${l}.sem \
                --data_words ${PMB_EXTDIR}/${l}/sents_${l}.sem \
                --data_chars ${PMB_EXTDIR}/${l}/sents_${l}_chars.txt \
                --word_embeddings ${FIT_WEMB} \
                --char_embeddings ${EMB_ROOT}/${l}/chars_${l}.txt \
                --use_words ${EMB_USE_WORDS} \
                --use_chars ${EMB_USE_CHARS} \
                --output ${MODEL_GIVEN_PATH} \
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
                --grid_search ${RUN_GRID_SEARCH} \
                --max_len_perc ${RUN_LEN_PERC} \
                --multi_word ${RUN_MWE}
    fi
done

