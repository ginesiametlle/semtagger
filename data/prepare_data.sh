#!/bin/bash
# this script prepares the data needed to train/test a sem-tagger


# base location of the Parallel Meaning Bank (PMB)
PMB_ROOT=${DATADIR}

# version of the Parallel Meaning Bank (PMB) to use (e.g. "pmb-1.0.0")
PMB_VER="pmb-1.0.0"

# languages from which to extract sem-tags
PMB_LANGS=("en")


# download the PMB
echo '[INFO] Downloading the PMB...'
PMB_HOME=${PMB_ROOT}/${PMB_VER}
if [ ! -d ${PMB_HOME} ] || [ ${newdata} -ge 1 ]; then
    rm -rf ${PMB_HOME}
    pushd ${PMB_ROOT} > /dev/null
    wget -q "pmb.let.rug.nl/releases/${PMB_VER}.zip"
    unzip "${PMB_VER}.zip"
    rm -f "${PMB_VER}.zip"
    popd > /dev/null
fi
echo '[INFO] Finished downloading the PMB'

# extract sem-tag data from the PMB
echo '[INFO] Extracting tag data from the PMB...'
POSDATA_HOME=${DATADIR}/pos
SEMDATA_HOME=${DATADIR}/sem

if [ ! -d ${POSDATA_HOME} ] || [ ! -d ${SEMDATA_HOME} ] || [ ${newdata} -ge 1 ]; then
    rm -rf ${POSDATA_HOME}
    mkdir -p ${POSDATA_HOME}
    rm -rf ${SEMDATA_HOME}
    mkdir -p ${SEMDATA_HOME}
    # iterate over p-parts in the PMB
    numsents=0
    for pdir in ${PMB_HOME}/data/* ; do
        # iterate over d-parts in p-parts
        for ddir in ${pdir}/* ; do
			      for l in ${PMB_LANGS[@]} ; do
				        if [ -f ${ddir}/${l}.drs.xml ]; then
            		    python3 ${UTILDIR}/extract_tags.py ${ddir}/${l}.drs.xml \
                            ${POSDATA_HOME}/pmb_${l}.txt \
                            ${SEMDATA_HOME}/pmb_${l}.txt
                    # feedback output
                    numsents=$((${numsents} + 1))
                    if ! ((${numsents} % 200)) && [ ${numsents} -ge 200 ] ; then
                        echo "[INFO] Processed ${numsents} sentences..."
                    fi
				        fi
			      done
        done
    done
fi
echo '[INFO] Extraction of sem-tag data completed'

