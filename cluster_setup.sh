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
pip install --user sregistry[registry] == 0.0.69
pip install --user git+https://github.com/bouthilx/flow.git

if [ "$CLUSTER" = "cedar" ]
then

cat << EOF
Add this into your .bashrc, taking care of replacing all <values>.

\`\`\`
module load singularity/2.5
module load 'nixpkgs/16.09' 'python/3.5'
\`\`\`
EOF

    # Do nothing
elif [ "$CLUSTER" = "graham" ]
then
    # Do nothing
else
    echo "unknown cluster; trying to install singularity (will fail if user is not root)"
    ./install_singularity.sh 2.5.2 /usr/local
    exit 1
fi

cat << EOF
Add this into your .bashrc, taking care of replacing all <values>.

\`\`\`
SINGULARITY_DIR=</some/path/to/singularity>
export SREGISTRY_STORAGE=\$SINGULARITY_DIR
export SINGULARITY_CACHEDIR=\$SINGULARITY_DIR/cache
export SREGISTRY_DATABASE=\$SINGULARITY_DIR
export SREGISTRY_NVIDIA_TOKEN=<SECRET>

export CONTAINER_DATA=</some/path/to/data>
export CONTAINER_HOME=</some/path/to/container/home>
export CONTAINER_CONFIG=</some/path/to/container/.config>
\`\`\`

If you use mongodb with a certificate file, also add this in your .bashrc

\`\`\`
export CERTIFICATE_FOLDER=</some/path/to/certs>
\`\`\`


Maybe also consider

\`\`\`
export PYTHONUNBUFFERED=1
\`\`\`
EOF

read -n 1 -s -r -p "Press any key to continue when done"

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

Set Kleio configs in 

\${CONTAINER_CONFIG}/kleio.core/kleio_config.yaml

Same as for Orion
EOF

read -n 1 -s -r -p "Press any key to continue when done"


# cd ${INSTALL_DIR}
    # git clone https://github.com/singularityhub/sregistry-cli
    # cd sregistry-cli
    #     git checkout tags/0.0.69 -b v0.0.69
    #     python setup.py install
    # cd ..
# cd ..
