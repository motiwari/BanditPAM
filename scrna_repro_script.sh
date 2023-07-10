#!/bin/bash

# 1. Install a ScRNA dataset
if [ -f "data/scrna_reformat.csv" ]; then
    echo "ScRNA found"
else
    echo "Installing ScRNA..."
    wget -P data https://motiwari.com/banditpam_data/scrna_reformat.csv.gz
    gunzip -k -c data/scrna_reformat.csv.gz > data/scrna_reformat.csv
    rm data/scrna_reformat.csv.gz
fi

# 2. Run the experiments on ScRNA
python experiments/run_scaling_experiment.py