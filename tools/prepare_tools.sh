#!/bin/bash
# this script downloads and installs external tools


# download and install the Elephant tokenizer
echo '[INFO] Preparing the Elephant tokenizer...'
if [ ! -d "${ELEPHANT_DIR}" ] || [ ! ${GET_TOOLS} -eq 0 ]; then
    rm -rf ${ELEPHANT_DIR}
    mkdir -p ${ELEPHANT_DIR}
    pushd ${ELEPHANT_DIR} > /dev/null
    wget -q "https://github.com/hslh/elephant/archive/master.zip"
    unzip -qq "master.zip"
    rm -f "master.zip"
    mv elephant-master/* .
    rm -rf elephant-master
    make
    popd > /dev/null
fi

