#!/bin/bash
apt install python

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
rm get-pip.py

pip install numpy
pip install networkx
