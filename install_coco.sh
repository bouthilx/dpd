#!/usr/bin/env bash

pip install numpy scipy

tmp=$(mktemp -d)

git clone --depth 1 https://github.com/numbbo/coco.git $tmp

cd $tmp
python do.py run-python

cd
rm -rf $tmp
