#!/usr/bin/env bash

pip install numpy scipy

cd ..
  git clone https://github.com/numbbo/coco.git
cd -
cd ../coco
  python do.py run-python
cd -
