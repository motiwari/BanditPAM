#!/bin/bash

# 1. Install BanditPAM
sudo python3 -m pip uninstall -y banditpam && sudo python3 -m pip install --no-use-pep517 -vvvv -e .

## 2. Install datasets if necessary
## MNIST
#if [ -d "data/MNIST_70k.csv" ]; then
#    echo "MNIST found"
#else
#    echo "Installing MNIST..."
#    wget -P data https://motiwari.com/banditpam_data/MNIST_70k.tar.gz
#    tar -xzvf data/MNIST_70k.tar.gz -C data
#    rm data/MNIST_70k.tar.gz
#fi

# 3. Run the experiments
python3 experiments/run_scaling_experiment.py

# 4. Plot the graph
python3 scripts/plot_graph.py