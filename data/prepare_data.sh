#!/bin/bash
# this script prepares the data needed to train and test a universal semantic tagger


# download the PMB Universal Semantic Tags release
echo "[INFO] Downloading the PMB Universal Semantic Tags release ${PMB_VER}..."
if [ ! -d ${PMB_ROOT} ] || [ ! ${GET_PMB} -eq 0 ] && [ ! ${PMB_MAIN_DATA} -eq 0 ]; then
    rm -rf ${PMB_ROOT}
    mkdir -p ${PMB_ROOT}
    pushd ${PMB_ROOT} > /dev/null
    wget -q "pmb.let.rug.nl/releases/sem-${PMB_VER}.zip"
    unzip -qq "sem-${PMB_VER}.zip"
    rm -f "sem-${PMB_VER}.zip"
    mv sem-${PMB_VER}/* .
    rm -rf "sem-${PMB_VER}"
    popd > /dev/null
fi
echo "[INFO] Finished downloading PMB data"

# extract semantic tags from sentences in the PMB
echo "[INFO] Extracting PMB data..."
for l in ${PMB_LANGS[@]} ; do
    if [ ! -f ${PMB_EXTDIR}/${l}/sem_${l}.sem ] || [ ! ${GET_PMB} -eq 0 ] && [ ! ${PMB_MAIN_DATA} -eq 0 ]; then
        rm -f ${PMB_EXTDIR}/${l}/sem_${l}.sem
        mkdir -p ${PMB_EXTDIR}/${l}
        numfiles=0
        # currently only English is available
        if [ ${l} == "en" ]; then
            # determine data locations according to PMB version
            pmb_data_sources=(${PMB_ROOT}/data/gold/* ${PMB_ROOT}/data/silver/*)
            # iterate over files in the PMB
            for pmb_data_source in ${pmb_data_sources[@]} ; do
                for pmb_file in ${pmb_data_source} ; do
                    # add file contents to existing data
                    awk 'BEGIN{ FS="\t" } { if ( NF > 1 ) print $1 "\t" $2 ; else print "" } END{ print "" }' ${pmb_file} \
                        >> ${PMB_EXTDIR}/${l}/sem_${l}.sem
                    # feedback output
                    numfiles=$((${numfiles} + 1))
                    if ! ((${numfiles} % 10000)) && [ ${numfiles} -ge 10000 ]; then
                        echo "[INFO] Processed ${numfiles} files (${l})..."
                    fi
                done
            done
        else
            echo -e "NIL\t.\n" >> ${PMB_EXTDIR}/${l}/sem_${l}.sem
        fi
        echo "[INFO] Extracted PMB data from ${numfiles} files (${l})"
    else
        rm -f ${PMB_EXTDIR}/${l}/sem_${l}.sem
        mkdir -p ${PMB_EXTDIR}/${l}
        echo -e "NIL\t.\n" >> ${PMB_EXTDIR}/${l}/sem_${l}.sem
    fi
done


# extract semantic tags from the extra available data
echo "[INFO] Extracting EXTRA data..."
if [ ! ${PMB_EXTRA_DATA} -eq 0 ]; then
    # remove remaining temporary files
    for l in ${PMB_LANGS[@]} ; do
        rm -f ${PMB_EXTDIR}/${l}/extra_${l}.sem.tmp
    done

    for idx in ${!PMB_EXTRA_LANGS[*]} ; do
        l=${PMB_EXTRA_LANGS[$idx]}
        if [ ! -f ${PMB_EXTDIR}/${l}/extra_${l}.sem ] || [ ! ${GET_EXTRA} -eq 0 ]; then
            rm -f ${PMB_EXTDIR}/${l}/extra_${l}.sem
            mkdir -p ${PMB_EXTDIR}/${l}
            numfiles=0
            # iterate over files in the extra directory
            for srcfile in ${PMB_EXTRA_SRC[$idx]}/* ; do
                # add file contents to existing data
                awk 'BEGIN{ FS="\t" } { if ( NF > 1 ) print $1 "\t" $2 ; else print "" } END{ print "" }' ${srcfile} \
                    >> ${PMB_EXTDIR}/${l}/extra_${l}.sem.tmp
                # feedback output
                numfiles=$((${numfiles} + 1))
                if ! ((${numfiles} % 10000)) && [ ${numfiles} -ge 10000 ] ; then
                    echo "[INFO] Processed ${numfiles} files (${l})..."
                fi
            done
            echo "[INFO] Extracted EXTRA data from ${numfiles} files (${l})"
        fi
    done
    for l in ${PMB_LANGS[@]} ; do
        if [ -f ${PMB_EXTDIR}/${l}/extra_${l}.sem.tmp ]; then
            mv -f ${PMB_EXTDIR}/${l}/extra_${l}.sem.tmp ${PMB_EXTDIR}/${l}/extra_${l}.sem
        fi
    done
else
    # ensure removal of remaining additional data
    for l in ${PMB_LANGS[@]} ; do
        rm -f ${PMB_EXTDIR}/${l}/extra_${l}.sem
        rm -f ${PMB_EXTDIR}/${l}/extra_${l}.sem.tmp
    done
fi
echo "[INFO] Extraction of sem-tag data completed"


# extract word embeddings or use pretrained ones
echo "[INFO] Preparing word embeddings..."
for idx in ${!PMB_LANGS[*]} ; do
    l=${PMB_LANGS[$idx]}
    lwpretrained=${EMB_WORD_PRETRAINED[$idx]}

    # only generate word embeddings when there are no pretrained ones
    if [ ! -f ${lwpretrained} ] || [ -z ${lwpretrained} ]; then
        # use GloVe embeddings for English
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
                wget -q "nlp.stanford.edu/data/${GLOVE_LINK}.zip"
                unzip -qq "${GLOVE_LINK}.zip"
                rm -f "${GLOVE_LINK}.zip"
                find . ! -name "${EMB_GLOVE_MODEL}.txt" -type f -exec rm -f {} +
                mv "${EMB_GLOVE_MODEL}.txt" "wemb_${l}.txt"
                popd > /dev/null
            fi
        # use Polyglot embeddings for languages other than English
        else
            echo "[INFO] Obtaining Polyglot embeddings for ${l}..."
            EMB_ROOT_LANG=${EMB_ROOT}/${l}
            if [ ! -d ${EMB_ROOT_LANG} ] || [ ! -f ${EMB_ROOT_LANG}/wemb_${l}.txt ] || [ ! ${GET_EMBS} -eq 0 ]; then
                rm -f ${EMB_ROOT_LANG}/wemb_${l}.txt
                mkdir -p ${EMB_ROOT_LANG}
                pushd ${EMB_ROOT_LANG} > /dev/null
                if [ ${l} == "de" ]; then
                    curl -L -s -o polyglot-${l} \
                         "https://docs.google.com/uc?id=0B5lWReQPSvmGaXJoQnlJa2x5RUU&export=download"
                elif [ ${l} == "it" ]; then
                    curl -L -s -o polyglot-${l} \
                         "https://docs.google.com/uc?id=0B5lWReQPSvmGM2gwSVdQVF9EOEk&export=download"
                elif [ ${l} == "nl" ]; then
                    curl -L -s -o polyglot-${l} \
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
        # use Gaussian initialization
        echo "[INFO] Initializing character embedding vectors for ${l}..."
        EMB_ROOT_LANG=${EMB_ROOT}/${l}
        if [ ! -d ${EMB_ROOT_LANG} ] || [ ! -f ${EMB_ROOT_LANG}/cemb_${l}.txt ] || [ ! ${GET_EMBS} -eq 0 ]; then
            rm -f ${EMB_ROOT_LANG}/cemb_${l}.txt
            mkdir -p ${EMB_ROOT_LANG}
            pushd ${EMB_ROOT_LANG} > /dev/null
            if [ ! -f ${PMB_EXTDIR}/${l}/extra_${l}.sem ]; then
                wchars=$(cat ${PMB_EXTDIR}/${l}/sem_${l}.sem | \
                                sed 's/./&\n/g' | LC_COLLATE=C sort -u | tr -d '\n')
            else
                wchars=$(cat ${PMB_EXTDIR}/${l}/sem_${l}.sem ${PMB_EXTDIR}/${l}/extra_${l}.sem | \
                                sed 's/./&\n/g' | LC_COLLATE=C sort -u | tr -d '\n')
            fi
            python3 ${DIR_DATA}/get_random_cemb.py $l ${wchars} ${EMB_ROOT_LANG}/cemb_${l}.txt
            popd > /dev/null
        fi
    fi
done
echo "[INFO] Finished preparing character embeddings"

