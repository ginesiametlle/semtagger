#!/bin/bash
# this script uses a trained semantic tagger on unlabeled data


# employ a model for each target language
for l in ${PMB_LANGS[@]} ; do

    MODEL_GIVEN_PATH="${MODEL_GIVEN_DIR}/${l}/tagger.hdf5"
    MODEL_PATH_INFO="${MODEL_GIVEN_DIR}/${l}/tagger_params.pkl"

    # use an existing model if it exists
    if [ -f ${MODEL_GIVEN_PATH} ] && [ ${GET_MODEL} -eq 0 ]; then
        echo "[INFO] A matching trained model was found for '${l}'"
        echo "[INFO] Using the model in ${MODEL_GIVEN_PATH}"
        python3 ${DIR_MODELS}/semtagger_predict.py ${DIR_ROOT} \
                --output_model ${MODEL_GIVEN_PATH} \
                --output_model_info ${MODEL_PATH_INFO} \
                --input_pred_file ${PRED_INPUT} \
                --output_pred_file ${PRED_OUTPUT}
    else
        echo "[INFO] No matching trained model was found for '${l}'"
    fi
done

