#!/bin/bash
# this script prepares the data needed to train/test a sem-tagger


# download the PMB
echo "[INFO] Downloading the PMB (version ${PMB_VER})..."
if [ ! -d ${PMB_ROOT} ] || [ ! ${GET_PMB} -eq 0 ]; then
    rm -rf ${PMB_ROOT}
    mkdir -p ${PMB_ROOT}
    pushd ${PMB_ROOT} > /dev/null
    wget -q "pmb.let.rug.nl/releases/pmb-${PMB_VER}.zip"
    unzip -qq "pmb-${PMB_VER}.zip"
    rm -f "pmb-${PMB_VER}.zip"
    mv pmb-${PMB_VER}/* .
    rm -rf "pmb-${PMB_VER}"
    popd > /dev/null
fi
echo '[INFO] Finished downloading the PMB'


# extract semantic tags from sentences in the PMB
echo '[INFO] Extracting tag data from the PMB...'
for l in ${PMB_LANGS[@]} ; do
    if [ ! -d ${PMB_EXTDIR}/${l} ] || [ ! ${GET_PMB} -eq 0 ]; then
    	  rm -rf ${PMB_EXTDIR}/${l}
    	  mkdir -p ${PMB_EXTDIR}/${l}
	      # iterate over p-parts in the PMB
	      numsents=0
	      for pdir in ${PMB_ROOT}/data/* ; do
	          # iterate over d-parts in p-parts
	          for ddir in ${pdir}/* ; do
                if [ -f ${ddir}/${l}.drs.xml ]; then
                    python3 ${DIR_UTILS}/extract_pmb_tags.py ${ddir}/${l}.drs.xml \
                            ${PMB_EXTDIR}/${l}/pmb_${l}.sem
                    # feedback output
                    numsents=$((${numsents} + 1))
                    if ! ((${numsents} % 200)) && [ ${numsents} -ge 200 ] ; then
                        echo "[INFO] Processed ${numsents} sentences..."
                    fi
                fi
            done
        done
        echo "[INFO] Extracted data contains ${numsents} sentences"
    fi
done
echo '[INFO] Extraction of tag data completed'


# download glove embeddings
echo "[INFO] Downloading ${GLOVE_MODEL} word embeddings..."
if [ ! -d ${GLOVE_ROOT} ] || [ ! -f ${GLOVE_ROOT}/${GLOVE_MODEL}.txt ] || [ ! ${GET_EMBS} -eq 0 ]; then
    rm -rf ${GLOVE_ROOT}
    mkdir -p ${GLOVE_ROOT}
    pushd ${GLOVE_ROOT} > /dev/null
    if [[ ${GLOVE_MODEL:0:8} = "glove.6B" ]]; then
        GLOVE_LINK="glove.6B"
    else
        GLOVE_LINK=${GLOVE_MODEL}
    fi
    wget -q "nlp.stanford.edu/data/${GLOVE_LINK}.zip"
    unzip -qq "${GLOVE_LINK}.zip"
    rm -f "${GLOVE_LINK}.zip"
    find . ! -name "${GLOVE_MODEL}.txt" -type f -exec rm -f {} +
    popd > /dev/null
fi
echo '[INFO] Finished downloading word embeddings'

