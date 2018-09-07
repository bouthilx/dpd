#!/usr/bin/env bash

if [ $EUID -ne 0 ]
then
   echo "This script must be run as root" 
   exit 1
fi

VERSION=$1
PREFIX=$2

if [ -f /etc/lsb-release ]
then
    apt-get install libarchive-dev
else
    echo "You must install libarchive-devel manually"
    read -n 1 -s -r -p "Press any key to continue when libarchive-devel installation is done"
fi

if [ -z "${INSTALL_DIR}" ]
then
    INSTALL_DIR=$HOME/repos
fi

if [ ! -d "${INSTALL_DIR}" ]
then
    mkdir -p ${INSTALL_DIR}
fi

cd ${INSTALL_DIR}
    git clone https://github.com/singularityware/singularity.git
    cd singularity
        git fetch --all
        git checkout $VERSION
        ./autogen.sh
        ./configure --prefix=$PREFIX
        make
        sudo make install
    cd ..
    rm singularity
cd ..
