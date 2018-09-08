#!/usr/bin/env bash

# if [ -z "${INSTALL_DIR}" ]
# then
#     INSTALL_DIR=$HOME/repos
# fi
# 
# if [ ! -d "${INSTALL_DIR}" ]
# then
#     mkdir -p ${INSTALL_DIR}
# fi
# 
if [ "$CLUSTER" = "cedar" ]
then
    # Do nothing
elif [ "$CLUSTER" = "graham" ]
then
    # Do nothing
else
    echo "unknown cluster; trying to install singularity (will fail if user is not root)"
    ./install_singularity.sh 2.5.2 /usr/local

    pip install --user sregistry[registry] == 0.0.69
    exit 1
fi


if [ ! -f $HOME/.sgdspacerc ]; then
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
    cp $DIR/.sgdspacerc $HOME/.sgdspacerc
    echo "Added \$HOME/.sgdspacerc"
fi

if grep -Fxq "source .sgdspacerc" $HOME/.bashrc 
then
    echo "bashrc already contains source .sgdspacerc"
else
    echo "source .sgdspacerc" >> $HOME/.bashrc
    echo "Added source .sgdspacerc to \$HOME/.bashrc"
fi

echo "Sourcing $HOME/.sgdspacerc"
source $HOME/.sgdspacerc

for path in SREGISTRY_STORAGE SINGULARITY_CACHEDIR SREGISTRY_DATABASE CONTAINER_DATA CERTIFICATE_FOLDER CONTAINER_HOME CONTAINER_CONFIG
do

    if [ ! -d "$path" ]
    then
        echo "creating $path"
        mkdir -p $path
    fi

done

echo "\n\n"

cat << EOF
Set Orion configs in 

\${HOME}/.config/orion.core/orion_config.yaml

Something like

\`\`\`
database:
    type: 'mongodb'
    name: '<DATABASE>'
    host: 'mongodb://<USER>:<PASSWORD>@<HOST>:<PORT>/<DATABASE>?ssl=true&ssl_ca_certs=<CERTIFICATE_PATH>&authSource=<USER>'
\`\`\`

EOF

read -n 1 -s -r -p "Press any key to continue when done"

mkdir ${HOME}/.config/kleio.core
cp ${HOME}/.config/orion.core/orion_config.yaml ${HOME}/.config/kleio.core/kleio_config.yaml

cat << EOF
Set Orion configs in

\${CONTAINER_CONFIG}/orion.core/orion_config.yaml

Something like

\`\`\`
database:
    type: 'mongodb'
    name: '<DATABASE>'
    host: 'mongodb://<USER>:<PASSWORD>@<HOST>:<PORT>/<DATABASE>?ssl=true&ssl_ca_certs=/certs/<CERTIFICATE_NAME>&authSource=<USER>'
\`\`\`

!! Note that the ssl_ca_certs should start with /certs/ as this is the expected path from within the
container
EOF

read -n 1 -s -r -p "Press any key to continue when done"

mkdir ${CONTAINER_CONFIG}/kleio.core
cp ${CONTAINER_CONFIG}/orion.core/orion_config.yaml ${CONTAINER_CONFIG}/kleio.core/kleio_config.yaml

pip install --user git+https://github.com/bouthilx/flow.git

# cd ${INSTALL_DIR}
    # git clone https://github.com/singularityhub/sregistry-cli
    # cd sregistry-cli
    #     git checkout tags/0.0.69 -b v0.0.69
    #     python setup.py install
    # cd ..
# cd ..
