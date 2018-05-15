#!/bin/bash
# this script prepares the data needed to train/test a sem-tagger


# download the PMB
echo "[INFO] Downloading the PMB (version ${PMB_VER})..."
if [ ! -d ${PMB_ROOT} ] || [ ! ${GET_PMB} -eq 0 ]; then
    rm -rf ${PMB_ROOT}
    mkdir -p ${PMB_ROOT}
    pushd ${PMB_ROOT} > /dev/null
    wget -q --show-progress "pmb.let.rug.nl/releases/pmb-${PMB_VER}.zip"
    unzip -qq "pmb-${PMB_VER}.zip"
    rm -f "pmb-${PMB_VER}.zip"
    mv pmb-${PMB_VER}/* .
    rm -rf "pmb-${PMB_VER}"
    popd > /dev/null
fi
echo "[INFO] Finished downloading the PMB"


# extract semantic tags from sentences in the PMB
echo "[INFO] Extracting tag data from the PMB..."
for l in ${PMB_LANGS[@]} ; do
    if [ ! -f ${PMB_EXTDIR}/${l}/pmb_${l}.sem ] || [ ! ${GET_PMB} -eq 0 ]; then
        rm -f ${PMB_EXTDIR}/${l}/pmb_${l}.sem
        mkdir -p ${PMB_EXTDIR}/${l}
        # determine data locations according to PMB version
        numsents=0
        if [ ${PMB_VER} == "1.0.0" ]; then
            pmb_data_sources=(${PMB_ROOT}/data/*)
        elif [ ${PMB_VER} == "2.0.0" ]; then
            pmb_data_sources=(${PMB_ROOT}/data/gold/* ${PMB_ROOT}/data/silver/*)
        fi
        # iterate over p-parts in the PMB
        for pmb_data_source in ${pmb_data_sources[@]} ; do
            for pdir in ${pmb_data_source} ; do
                # iterate over d-parts in p-parts
                for ddir in ${pdir}/* ; do
                    if [ -f ${ddir}/${l}.drs.xml ]; then
                        python3 ${DIR_DATA}/get_pmb_tags.py ${ddir}/${l}.drs.xml \
                                ${PMB_EXTDIR}/${l}/pmb_${l}.sem
                        # feedback output
                        numsents=$((${numsents} + 1))
                        if ! ((${numsents} % 1000)) && [ ${numsents} -ge 1000 ] ; then
                            echo "[INFO] Processed ${numsents} sentences (${l})..."
                        fi
                    fi
                done
            done
        done
        echo "[INFO] Extracted PMB data contains ${numsents} sentences (${l})"
    fi
done


# extract semantic tags from the extra available data
echo "[INFO] Extracting extra tag data..."
if [ ! ${PMB_EXTRA_DATA} -eq 0 ]; then
    for idx in ${!PMB_EXTRA_LANGS[*]} ; do
        l=${PMB_EXTRA_LANGS[$idx]}
        if [ ! -f ${PMB_EXTDIR}/${l}/extra_${l}.sem ] || [ ! ${GET_EXTRA} -eq 0 ]; then
            rm -f ${PMB_EXTDIR}/${l}/extra_${l}.sem
            mkdir -p ${PMB_EXTDIR}/${l}
            # iterate over files in the extra directory
            numsents=0
            for srcfile in ${PMB_EXTRA_SRC[$idx]}/* ; do
                # add file contents to existing data
                awk 'BEGIN{ FS="\t" } { print $2 "\t" $1 } END{ print "" }' ${srcfile} \
                    >> ${PMB_EXTDIR}/${l}/extra_${l}.sem
                # feedback output
                numsents=$((${numsents} + 1))
                if ! ((${numsents} % 1000)) && [ ${numsents} -ge 1000 ] ; then
                    echo "[INFO] Processed ${numsents} sentences (${l})..."
                fi
            done
            echo "[INFO] Extracted extra data contains ${numsents} sentences (${l})"
        fi
    done
else
    # ensure removal of remaining additional data
    for l in ${PMB_LANGS[@]} ; do
        rm -f ${PMB_EXTDIR}/${l}/extra_${l}.sem
    done
fi
echo "[INFO] Extraction of tag data completed"


# extract word embeddings or use pretrained ones
echo "[INFO] Preparing word embeddings..."
for idx in ${!PMB_LANGS[*]} ; do
    l=${PMB_LANGS[$idx]}
    lwpretrained=${EMB_WORD_PRETRAINED[$idx]}

    # only generate word embeddings when there are no pretrained ones
    if [ ! -f ${lwpretrained} ] || [ -z ${lwpretrained} ] && [ ! ${EMB_USE_WORDS} -eq 0 ]; then
        # use glove embeddings for english
        if [ ${l} == "en" ]; then
            echo "[INFO] Obtaining GloVe word embeddings for ${l}..."
            EMB_ROOT_EN=${EMB_ROOT}/${l}
            if [ ! -d ${EMB_ROOT_EN} ] || [ ! -f ${EMB_ROOT_EN}/wemb_${l}.txt ] || [ ! ${GET_EMBS} -eq 0 ]; then
                rm -f ${EMB_ROOT_EN}/wemb_${l}.txt
                mkdir -p ${EMB_ROOT_EN}
                pushd ${EMB_ROOT_EN} > /dev/null
                if [[ ${EMB_GLOVE_MODEL:0:8} = "glove.6B" ]]; then
                    GLOVE_LINK="glove.6B"
                else
                    GLOVE_LINK=${EMB_GLOVE_MODEL}
                fi
                wget -q --show-progress "nlp.stanford.edu/data/${GLOVE_LINK}.zip"
                unzip -qq "${GLOVE_LINK}.zip"
                rm -f "${GLOVE_LINK}.zip"
                find . ! -name "${EMB_GLOVE_MODEL}.txt" -type f -exec rm -f {} +
                mv "${EMB_GLOVE_MODEL}.txt" "wemb_${l}.txt"
                popd > /dev/null
            fi
        # use polyglot embeddings for languages other than english
        else
            echo "[INFO] Obtaining Polyglot embeddings for ${l}..."
            EMB_ROOT_LANG=${EMB_ROOT}/${l}
            if [ ! -d ${EMB_ROOT_LANG} ] || [ ! -f ${EMB_ROOT_LANG}/wemb_${l}.txt ] || [ ! ${GET_EMBS} -eq 0 ]; then
                rm -f ${EMB_ROOT_LANG}/wemb_${l}.txt
                mkdir -p ${EMB_ROOT_LANG}
                pushd ${EMB_ROOT_LANG} > /dev/null
                if [ ${l} == "de" ]; then
                    curl -L -s -o --progress-bar polyglot-${l} \
                         "https://docs.google.com/uc?id=0B5lWReQPSvmGaXJoQnlJa2x5RUU&export=download"
                elif [ ${l} == "it" ]; then
                    curl -L -s -o polyglot-${l} \
                         "https://docs.google.com/uc?id=0B5lWReQPSvmGM2gwSVdQVF9EOEk&export=download"
                elif [ ${l} == "nl" ]; then
                    curl -L -s -o --progress-bar polyglot-${l} \
                         "https://docs.google.com/uc?id=0B5lWReQPSvmGNUprVTVNY3I3eDA&export=download"
                else
                    echo "[ERROR] Language ${l} does not appear to be a language in the PMB"
                    exit
                fi
                python3 ${DIR_DATA}/get_polyglot_wemb.py ${l} ${EMB_ROOT_LANG}/polyglot-${l} ${EMB_ROOT_LANG}/wemb_${l}.txt
                rm -f polyglot-${l}
                popd > /dev/null
            fi
        fi
    fi
done
echo "[INFO] Finished preparing word embeddings"


# extract character embeddings or use pretrained ones
echo "[INFO] Preparing character embeddings..."
for idx in ${!PMB_LANGS[*]} ; do
    l=${PMB_LANGS[$idx]}
    lcpretrained=${EMB_CHAR_PRETRAINED[$idx]}
    # only generate character embeddings when there are no pretrained ones
    if [ ! -f ${lcpretrained} ] || [ -z ${lcpretrained} ] && [ ! ${EMB_USE_CHARS} -eq 0 ]; then
        # use lm_1b model for English
        if [ ${l} == "en" ]; then
            echo "[INFO] Obtaining lm_1b character embeddings for ${l}..."
            EMB_ROOT_EN=${EMB_ROOT}/${l}
            if [ ! -d ${EMB_ROOT_EN} ] || [ ! -f ${EMB_ROOT_EN}/cemb_${l}.txt ] || [ ! ${GET_EMBS} -eq 0 ]; then
                rm -f ${EMB_ROOT_EN}/cemb_${l}.txt
                mkdir -p ${EMB_ROOT_EN}
                pushd ${EMB_ROOT_EN} > /dev/null
                pushd ${LM1B_DIR} > /dev/null
                bazel-bin/lm_1b/lm_1b_eval --mode dump_emb --pbtxt data/graph-2016-09-10.pbtxt \
                                           --vocab_file data/vocab-small.txt \
                                           --ckpt 'data/ckpt-*' --save_dir output
                popd > /dev/null
                python3 ${DIR_DATA}/get_lm1b_cemb.py $l ${LM1B_DIR}/output/embeddings_char_cnn.npy ${EMB_ROOT_EN}/cemb_${l}.txt
                popd > /dev/null
            fi
        # use Gaussian initialization on other languages
        else
            echo "[INFO] Generating character embedding for ${l}..."
            EMB_ROOT_LANG=${EMB_ROOT}/${l}
            if [ ! -d ${EMB_ROOT_LANG} ] || [ ! -f ${EMB_ROOT_LANG}/cemb_${l}.txt ] || [ ! ${GET_EMBS} -eq 0 ]; then
                rm -f ${EMB_ROOT_LANG}/cemb_${l}.txt
                mkdir -p ${EMB_ROOT_LANG}
                pushd ${EMB_ROOT_LANG} > /dev/null
                wchars=$(cat ${PMB_EXTDIR}/${l}/pmb_${l}.sem | \
                                sed 's/./&\n/g' | LC_COLLATE=C sort -u | tr -d '\n')
                python3 ${DIR_DATA}/get_random_cemb.py $l ${wchars} ${EMB_ROOT_LANG}/cemb_${l}.txt
                popd > /dev/null
            fi
        fi
    fi
done
echo "[INFO] Finished preparing character embeddings"

