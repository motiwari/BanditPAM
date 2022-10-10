# instsall requirements
git submodule update --init --recursive
cd headers/carma
mkdir build && cd build && cmake -DCARMA_INSTALL_LIB=ON .. && sudo cmake --build . --config Release --target install
cd ../../..
pip install -r requirements.txt
sudo pip install .

# install dataset
wget https://motiwari.com/banditpam_data/MNIST_70k.tar.gz -P data
tar -xf data/MNIST_70k.tar.gz -C data

# run default experiments (n_medoids=[5, 10], n_data=[10000, 30000])
python scripts/experiment.py