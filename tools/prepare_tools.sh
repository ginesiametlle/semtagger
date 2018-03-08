#!/bin/bash
# this script downloads and installs external tools


# download and install the Elephant tokenizer
ELEPHANT_DIR="${DIR_TOOLS}/elephant-master"
echo '[INFO] Preparing the Elephant tokenizer...'
if [ ! -d "${ELEPHANT_DIR}" ] || [ ${GET_TOOLS} -ge 1 ]; then
    rm -rf ${ELEPHANT_DIR}
    pushd ${DIR_TOOLS} > /dev/null
    wget -q "https://github.com/hslh/elephant/archive/master.zip"
    unzip "master.zip"
    rm -f "master.zip"
    popd > /dev/null
    pushd ${ELEPHANT_DIR} > /dev/null
    make
    popd > /dev/null
fi

