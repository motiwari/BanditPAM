#!/bin/bash
######### Helper Functions #########
download () {
  url=$1
  filename=$2

  # if [ -x "$(which wget)" ] ; then
  #     wget -O $filename $url
  if [ -x "$(which curl)" ]; then
      curl -o $filename -sfL $url -v
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

######### Run scripts #########
sh ./build_docker.sh
sh scripts/install-hooks.sh
