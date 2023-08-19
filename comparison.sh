#!/bin/bash

# 1. Install BanditPAM
python -m pip install -r requirements.txt
python -m pip install banditpam

# 2. Install datasets if necessary
# MNIST
if [ -d "data/MNIST_70k.csv" ]; then
    echo "MNIST found"
else
    echo "Installing MNIST..."
    wget -P data https://motiwari.com/banditpam_data/MNIST_70k.tar.gz
    tar -xzvf data/MNIST_70k.tar.gz -C data
    rm data/MNIST_70k.tar.gz
fi

# scRNA
if [ -d "data/scrna_reformat.csv" ]; then
    echo "scRNA found"
else
    echo "Installing scRNA..."
    wget -P data https://motiwari.com/banditpam_data/scrna_reformat.csv.gz
    gunzip data/scrna_reformat.csv.gz
fi

# 3. Run the experiments
python scripts/compare_banditpam_versions.py
