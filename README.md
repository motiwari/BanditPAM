# cooking_spray
C++ implmentation of PAM with confidence bounds.

installation
from main directory
```
cd build
cmake ..
make
```

for command line arguments
`./pam ../data/mnist.csv output_name 5`
will run the pam algorithim on the mnist toy set with 5 clusters. The output format is not currently implemented (but will just involve writing assignments and medoids to a file)
