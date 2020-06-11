# cooking_spray
C++ implmentation of PAM with confidence bounds.

## installation
### requirements
Building this repository requires three external requirements: 

Cmake >= 3.17

Armadillo >= 9.7

openMP >= 2.5

Armadillo is a C++ library for linear algebra and matrix operations, and OpenMP is 
a language/package used for multithreading.

Armadillo: http://arma.sourceforge.net/download.html
OpenMP: https://www.openmp.org/resources/openmp-compilers-tools/
OpenMP is supported by default on most Linux platforms, and can be downloaded through
homebrew on macs.

### building
The build process is automated with cmake. After cloning the repository the program
can be built with the following steps.
```
~/cooking_spray$ mkdir build
~/cooking_spray$ cd build
~/cooking_spray/build$ cmake ..
~/cooking_spray/build$ make
```

## usage
The easiest way to use the program is with the compiled `pam` binary. This is a command 
line program that takes in three arguments: the dataset, the output path, and the number 
of clusters to fit the data to.

To run the KMedoids algorithim on the MNIST dataset that is contained in a file called 
`mnist.csv` contained in a `data` folder, and we wish to fit our data to 5 clusters then
we could run the following command.
```
~/cooking_spray/build$ ./pam ../data/mnist.csv output_name 5
```

## future development
* should data be a parameter that is passed into the constructor?
* medoid_indices should be private class variable?
* test medoid matrix vs medoid indicies performance?
* To consider n - k for swaps -> change candidate vector?
* generic metric type
* error checking for number of clusters >= unique data points
* should the mediods be index numbers or the mediods themselves?
* switch all naming to camelCase
* most typedefs are 64 bits, overkill?
* check if N in build targets, then just iterate?
* should loss function be a template object?
* change name so can work with basic and ucb variation