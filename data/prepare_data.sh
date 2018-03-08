#!/bin/bash
# this script prepares the data needed to train/test a sem-tagger


# download the PMB
echo '[INFO] Downloading the PMB...'
if [ ! -d ${PMB_ROOT} ] || [ ${GET_PMB} -ge 1 ]; then
    rm -rf ${PMB_ROOT}
	mkdir -p ${PMB_ROOT}
    pushd ${PMB_ROOT} > /dev/null
    wget -q "pmb.let.rug.nl/releases/pmb-${PMB_VER}.zip"
    unzip "pmb-${PMB_VER}.zip"
    rm -f "pmb-${PMB_VER}.zip"
	mv pmb-${PMB_VER}/* .
	rm -rf "pmb-${PMB_VER}"
    popd > /dev/null
fi
echo '[INFO] Finished downloading the PMB'


# extract semantic tags from sentences in the PMB
echo '[INFO] Extracting tag data from the PMB...'
for l in ${PMB_LANGS[@]} ; do
    if [ ! -d ${DIR_DATA}/${l} ] || [ ${GET_PMB} -ge 1 ]; then
    	  rm -rf ${DIR_DATA}/${l}
    	  mkdir -p ${DIR_DATA}/${l}
	      # iterate over p-parts in the PMB
	      numsents=0
	      for pdir in ${PMB_ROOT}/data/* ; do
	          # iterate over d-parts in p-parts
	          for ddir in ${pdir}/* ; do
                if [ -f ${ddir}/${l}.drs.xml ]; then
                    python3 ${DIR_UTILS}/extract_tags.py ${ddir}/${l}.drs.xml \
                            ${DIR_DATA}/${l}/sents_${l}.sem
                    # feedback output
                    numsents=$((${numsents} + 1))
                    if ! ((${numsents} % 200)) && [ ${numsents} -ge 200 ] ; then
                        echo "[INFO] Processed ${numsents} sentences..."
                    fi
                fi
            done
        done
    fi
done
echo '[INFO] Extraction of tag data completed'


# download glove embeddings
echo '[INFO] Downloading ${GLOVE_MODEL} word embeddings...'
if [ ! -d ${GLOVE_ROOT} ] || [ ${GET_EMBS} -ge 1 ]; then
    rm -rf ${GLOVE_ROOT}
    mkdir -p ${GLOVE_ROOT}
    pushd ${GLOVE_ROOT} > /dev/null
    wget "nlp.stanford.edu/data/${GLOVE_MODEL}.zip"
    unzip "${GLOVE_MODEL}.zip"
    rm -f "${GLOVE_MODEL}.zip"
    popd > /dev/null
fi
echo '[INFO] Finished downloading embeddings'

