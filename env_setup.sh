######### Executables #########
chmod +x scripts/*
chmod +x build_docker.sh

######### Get data #########
mkdir -p data
#### Get SCRNA data
wget -P data http://web.stanford.edu/~ericsf/scrna_reformat.csv.gz
#### Get MNIST data
wget -P data http://web.stanford.edu/~ericsf/mnist.csv
wget -P data http://web.stanford.edu/~ericsf/MNIST-70k.csv
head -60000 data/MNIST-70k.csv > data/MNIST-60k.csv
head -65000 data/MNIST-70k.csv > data/MNIST-65k.csv
head -1000 data/MNIST-70k.csv > data/MNIST-1k.csv

######### Run scripts #########
sh ./build_docker.sh
sh ./install-hooks.sh
