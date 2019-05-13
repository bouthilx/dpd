./install_coco.sh
./install_mahler.sh

pip install --upgrade git+https://github.com/bouthilx/orion.git@4558e1b54de317b72ef21d61fc332b3754e6b1a3
pip install --upgrade git+https://github.com/bouthilx/orion.algo.optuna.git
pip install --upgrade git+https://github.com/Epistimio/orion.algo.skopt.git@de8fa3c

pip install -e '.[coco,visiondl,tpe,bayesopt,dpd]'
