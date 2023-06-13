#!/bin/bash

# 1. Install BanditPAM
pip install -r requirements.txt
pip install banditpam

# 2. Install datasets if necessary
# MNIST
if [ -f "data/MNIST_70k.csv" ]; then
    echo "MNIST found"
else
    echo "Installing MNIST..."
    wget -P data https://motiwari.com/banditpam_data/MNIST_70k.tar.gz
    tar -xzvf data/MNIST_70k.tar.gz -C data
    rm data/MNIST_70k.tar.gz
fi

# CIFAR-10
if [ -f "data/cifar10.csv" ]; then
    echo "CIFAR-10 found"
else
    echo "Installing CIFAR-10..."
    wget -P data https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xzvf data/cifar-10-python.tar.gz -C data
    rm data/cifar-10-python.tar.gz
    # Preprocess the dataset
    python data/preprocess_cifar.py
fi

# 3. Run the experiments
python experiments/run_scaling_experiment.py