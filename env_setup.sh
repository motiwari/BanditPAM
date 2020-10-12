######### Helper Functions #########
function download {
    url=$1
    filename=$2

    if [ -x "$(which wget)" ] ; then
        wget -q $url -O $2
    elif [ -x "$(which curl)" ]; then
        curl -o $2 -sfL $url
    else
        echo "Could not find curl or wget, please install one." >&2
    fi
}

######### Executables #########
chmod +x scripts/*
chmod +x build_docker.sh

######### Get data #########
mkdir -p data
#### Get SCRNA data
download http://web.stanford.edu/~ericsf/scrna_reformat.csv.gz ./data/scrna_reformat.csv.gz

#### Get MNIST data
download http://web.stanford.edu/~ericsf/mnist.csv ./data/MNIST.csv
download http://web.stanford.edu/~ericsf/MNIST-70k.csv ./data/MNIST-70k.csv
head -60000 data/MNIST-70k.csv > data/MNIST-60k.csv
head -65000 data/MNIST-70k.csv > data/MNIST-65k.csv
head -10000 data/MNIST-70k.csv > data/MNIST-10k.csv
head -20000 data/MNIST-70k.csv > data/MNIST-20k.csv
head -40000 data/MNIST-70k.csv > data/MNIST-40k.csv
head -1000 data/MNIST-70k.csv > data/MNIST-1k.csv

######### Run scripts #########
sh ./build_docker.sh
sh scripts/install-hooks.sh
