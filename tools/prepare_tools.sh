#!/bin/bash
# this script downloads and installs external tools


ELEPHANT_DIR="${TOOLDIR}/elephant-master"

# download and install the Elephant tokenizer
echo '[INFO] Preparing the Elephant tokenizer...'
if [ ! -d "${ELEPHANT_DIR}" ] || [ $newtools -ge 1 ]; then
    rm -rf ${ELEPHANT_DIR}
    pushd ${TOOLDIR} > /dev/null
    wget -q "https://github.com/hslh/elephant/archive/master.zip"
    unzip "master.zip"
    rm -f "master.zip"
    popd > /dev/null
    pushd ${ELEPHANT_DIR} > /dev/null
    make
    popd > /dev/null
fi

