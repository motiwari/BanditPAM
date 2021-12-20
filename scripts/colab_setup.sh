#!/bin/bash

apt-get update
apt install build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev 

# Should be in /content/
curl -XGET https://codeload.github.com/Kitware/CMake/tar.gz/refs/tags/v3.22.1 > cmake-v3.22.1.tar.gz
tar -xvzf cmake-v3.22.1.tar.gz
cd cmake-v3.22.1
mkdir build && cd build && cmake .. && make && make install

git clone https://github.com/ThrunGroup/BanditPAM.git
cd BanditPAM
git submodule update --init --recursive
cd headers/carma
mkdir build && cd build && cmake .. && make
cd ../extern/armadillo-code
mkdir build && cd build && cmake .. && make && make install

cd ../../../../..

pip install .
