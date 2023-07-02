# Installation Tutorial for Linux

The following is a more detailed description of the installation process of BanditPAM for Linux machines.

## Prerequisites
Please ensure the following dependencies are installed:
 - A package manager like `yum` or `apt`
 - A C++ compiler; we recommend LLVM's `clang`: via the [LLVM installation instructions](https://clang.llvm.org/get_started.html)
 - `CMake`: via the [CMake installation instructions](https://cmake.org/install/)
 - OpenMP: via `sudo apt install libomp-dev` or `sudo yum install libomp-dev`
 - Armadillo: via the [Armadillo installation instructions](http://arma.sourceforge.net/download.html)
 - CARMA: via the instructions in [its guide](https://github.com/RUrlus/carma#installation)
 - Python3: if not installed, we recommend installing Python3 via [Anaconda](https://www.anaconda.com/products/individual), which is CPython compiled with `clang`
 - `pip` for your Python3 installation: this should be completed if installing via Anaconda above
 - The necessary python packages: via `python -m pip install -r requirements.txt`
 
 (NOT RECOMMENDED): Instead of LLVM's `clang`, you can also use another C++ compiler and point your `CC` environment variable to it. Please ensure this is the same compiler used to compile your Python installation if using CPython. If you open a `python` REPL it will show the compiler used during the language installation:

 ```
 >> python
Python 3.7.9 (default, Mar  1 2021, 13:32:26)
[Clang 11.0.0 (clang-1100.0.33.17)] :: Intel Corporation on darwin
Type "help", "copyright", "credits" or "license" for more information.
Intel(R) Distribution for Python is brought to you by Intel Corporation.
Please check out: https://software.intel.com/en-us/python-distribution
```

Do not attempt to compile the BanditPAM extension with `gcc` if your Python used `clang`, and vice versa.

## BanditPAM Installation

BanditPAM can then be installed via one of the following ways:
1) Running `python -m pip install banditpam`, OR
2) Running `python -m pip install .` in the home directory (`/BanditPAM`)

## Known Issues 
The following is a list of issues seen when installing BanditPAM on Linux. To report a bug, please file an issue at https://github.com/motiwari/BanditPAM/

- Beware that installing `libarmadillo-dev` using `apt` or `yum` may provide an outdated version of `armadillo`, especially if you are running an older version of Linux. In this case, you may need to download the latest stable version of `armadillo` and compile it from source
