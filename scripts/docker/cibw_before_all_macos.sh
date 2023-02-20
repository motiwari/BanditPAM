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
sudo cmake --build . --config Release --target install