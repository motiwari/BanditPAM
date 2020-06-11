# BanditPAM
C++ implementation of BanditPAM with confidence bounds.

## Installation
### Requirements
Building this repository requires three external requirements: 
* Cmake >= 3.17
* Armadillo >= 9.7
* openMP >= 2.5

Armadillo is a C++ library for linear algebra and matrix operations, and OpenMP is 
a language/package used for multithreading.

* Armadillo: http://arma.sourceforge.net/download.html
* OpenMP: https://www.openmp.org/resources/openmp-compilers-tools/
OpenMP is supported by default on most Linux platforms, and can be downloaded through
homebrew on macs.

### Building
The build process is automated with cmake. After cloning the repository the program
can be built with the following steps.
```
/BanditPAM$ mkdir build
/BanditPAM$ cd build
/BanditPAM/build$ cmake ..
/BanditPAM/build$ make
```

## Usage
The easiest way to use the program is with the compiled `BanditPAM` binary. This is a command 
line program that takes in three arguments: the dataset, the number of clusters, and a parameter
indicating whether or not to output data point assignments. 

```
./BanditPAM -f [path/to/input.csv] -k [number of clusters] -a
```
* `-f` is mandatory, and has a mandatory argument that specifies the path to the dataset to load in.
* `-k` is mandatory, and specifies the number of clusters to fit the data to.
* `-a` is optional, and if this flag is specified then the cluster assignments will be printed to standard out

To run the KMedoids algorithim on the MNIST dataset that is contained in a file called 
`mnist.csv` contained in a `data` folder, and we wish to fit our data to 5 clusters then
we could run the following command.
```
/BanditPAM/build$ ./BanditPAM -f ../data/mnist.csv -k 5
```