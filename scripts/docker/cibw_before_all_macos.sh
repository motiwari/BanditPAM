export PATH="/usr/local/Cellar/gcc/12.2.0/bin:$PATH"
export CC="/usr/local/Cellar/gcc/12.2.0/bin/gcc-12"
export CXX="/usr/local/Cellar/gcc/12.2.0/bin/g++-12"
cd ~
git clone https://gitlab.com/conradsnicta/armadillo-code.git
git clone https://github.com/RUrlus/carma.git --recursive # Do we need this?
cd ~/armadillo-code
cmake .
make install
cd ~/carma
mkdir -p build
cd build
cmake -DCARMA_INSTALL_LIB=ON ..
sudo CC="/usr/local/Cellar/gcc/12.2.0/bin/gcc-12" CXX="/usr/local/Cellar/gcc/12.2.0/bin/g++-12" cmake --build . --config Release --target install