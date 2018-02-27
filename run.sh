#!/bin/bash
# this is a general setup script for this project


# update data with option --fetch-data, -d
newdata=0

# update existing tools with option --setup-tools, -t
newtools=0

# re-train the sem-tagger with option --sem-tag -s
newtagger=0

# set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands'
set -e
set -u
set -o pipefail
#set -x

# define directory locations

# root directory for this project
ROOTDIR=$PWD

# data directory
DATADIR=${ROOTDIR}/data

# tools directory
TOOLDIR=${ROOTDIR}/tools

# models directory
MODELDIR=${ROOTDIR}/models

# utils directory
UTILDIR=${ROOTDIR}/utils


# ensure script runs from the root directory
if ! [ -x "${ROOTDIR}/run.sh" ]; then
    echo '[INFO] You must run setup.sh from the root directory'
    exit 1
fi

# transform long options to short ones and parse them
for arg in "$@"; do
    shift
    case "$arg" in
        "--fetch-data") set -- "$@" "-d" ;;
        "--setup-tools") set -- "$@" "-t" ;;
        "--sem-tag") set -- "$@" "-s" ;;
        *) set -- "$@" "$arg"
    esac
done

while getopts s:dts option
do
    case "${option}"
    in
        d) newdata=1;;
        t) newtools=1;;
        s) newtagger=1;;
    esac
done


###############################
#  DOWNLOAD AND PREPARE DATA  #
###############################
echo '[INFO] Preparing data...'
. ${DATADIR}/prepare_data.sh
echo '[INFO] Finished preparing data'


##########################
#  SETUP REQUIRED TOOLS  #
##########################
echo '[INFO] Setting up required tools...'
. ${TOOLDIR}/prepare_tools.sh
echo '[INFO] Finished setting up tools'


########################################
#  TRAIN A MODEL FOR SEMANTIC TAGGING  #
########################################
echo '[INFO] Training a new sem-tagger...'
. ${MODELDIR}/prepare_semtagger.sh
echo '[INFO] A sem-tagger was succesfully trained'

