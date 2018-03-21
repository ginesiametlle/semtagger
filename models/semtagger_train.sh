#!/bin/bash
# this script trains a new semantic tagger on the input data


# employ a model for each target language
for l in ${PMB_LANGS[@]} ; do
    modelname=${MODEL_TYPE}-${MODEL_SIZE}-${MODEL_LAYERS}-${MODEL_ACTIVATION_OUTPUT}-${l}

    # use an existing model if it exists
    if [ -d ${MODEL_ROOT}/${modelname} ] && [ ${GET_MODEL} -eq 0 ]; then
        echo "[INFO] A matching trained model was found for '${l}'"
        echo "[INFO] Using the model in ${MODEL_ROOT}/${modelname} (see configuration file)"

    # train a new model
    else
        echo "[INFO] Training a new model for language '${l}'..."
        echo "[INFO] The new model will be stored in ${MODEL_ROOT}/${modelname}"
        rm -rf ${MODEL_ROOT}/${modelname}
        mkdir -p ${MODEL_ROOT}/${modelname}
        python3 ${DIR_MODELS}/semtagger_fit.py ${DIR_ROOT} \
                --raw_data ${PMB_EXTDIR}/${l}/pmb_${l}.sem \
                --data ${PMB_EXTDIR}/${l}/sents_${l}.sem \
                --embeddings ${GLOVE_ROOT}/${GLOVE_MODEL}.txt \
                --output ${MODEL_ROOT}/${modelname} \
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
                --learning_rate ${MODEL_LEARNING_RATE} \
                --dropout ${MODEL_DROPOUT} \
                --batch_size ${MODEL_BATCH_SIZE} \
                --batch_normalization ${MODEL_BATCH_NORMALIZATION} \
                --verbose ${MODEL_VERBOSE} \
                --test_size ${RUN_TEST_SIZE} \
                --cross_validate ${RUN_CROSS_VAL} \
                --max_len_perc ${RUN_LEN_PERC} \
                --multi_word ${RUN_MWE}
    fi
done

