pkgutil --pkg-info=com.apple.pkg.CLTools_Executables
xcodebuild -version
sudo xcode-select -s /Library/Developer/CommandLineTools
echo "LOOK HERE"
# sudo rm -rf /Library/Developer/CommandLineTools
# xcode-select --install
# xcodebuild -version
pip install numpy
gcc --version
g++ --version
cmake --version
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
#sudo find / -iname "carma"

# These do not work
# For a solution, see https://github.com/pypa/cibuildwheel/issues/816

# Brew install armadillo llvm libomp
export DYLD_LIBRARY_PATH=/usr/local/opt/armadillo/lib 
export LD_LIBRARY_PATH=/usr/local/opt/armadillo/lib


