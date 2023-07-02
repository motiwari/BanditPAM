#!/bin/bash
######### Helper Functions #########
download () {
  url=$1
  filename=$2

  if [ -x "$(which wget)" ] ; then
      wget -O $filename $url
  elif [ -x "$(which curl)" ]; then
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
[[ ! -f data/scrna_reformat.csv.gz ]] && echo "Downloading SCRNA..." && download https://motiwari.com/banditpam_data/scrna_reformat.csv.gz data/scrna_reformat.csv.gz

#### Get MNIST data
[[ ! -f data/MNIST.csv ]] && echo "Downloading small MNIST..." && download https://motiwari.com/banditpam_data/MNIST_100.csv data/MNIST_100.csv
[[ ! -f data/MNIST-70k.csv ]] && "Downloading full MNIST..." && download https://motiwari.com/banditpam_data/MNIST_70k.tar.gz data/MNIST_70k.tar.gz

######### Run scripts #########

if [[ "$(docker images -q banditpam/cpp:latest 2> /dev/null)" == "" ]]; then
  echo "Running build_docker.sh"
  sh ./build_docker.sh
else
  echo "Docker image banditpam/cpp:latest already exists. Skipping execution of build_docker.sh."
fi

echo "Done"
