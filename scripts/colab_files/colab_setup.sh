#!/bin/bash

apt-get update
apt install build-essential checkinstall libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev

# Should be in /content/
cd /content
curl -XGET https://codeload.github.com/Kitware/CMake/tar.gz/refs/tags/v3.22.1 > cmake-v3.22.1.tar.gz
tar -xvzf cmake-v3.22.1.tar.gz
cd /content/CMake-3.22.1 && mkdir build && cd build && cmake .. && make && make install

cd /content
git clone https://github.com/motiwari/BanditPAM.git
cd /content/BanditPAM
git submodule update --init --recursive
cd /content/BanditPAM/headers/carma
mkdir build && cd build && cmake .. && make
cd /content/BanditPAM/headers/carma/extern/armadillo-code
mkdir build && cd build && cmake .. && make && make install

pip install --no-cache-dir /content/BanditPAM
