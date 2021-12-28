# Installation Tutorial for Windows

The following is a description of the installation process of BanditPAM for Windows. This assumes that:
 
## Prerequisites
Please ensure the following dependencies are installed:
 - A C++ compiler; we recommend LLVM's `clang`: via the [LLVM installation instructions](https://clang.llvm.org/get_started.html)
 - OpenMP: if using LLVM's `clang`, then OpenMP is already enabled
 - `CMake`: via the [CMake installation instructions](https://cmake.org/install/)
 - Armadillo: via the [Armadillo installation instructions](http://arma.sourceforge.net/download.html)
 - CARMA: via the instructions in the [quickstart](https://github.com/ThrunGroup/BanditPAM#install-the-repo-and-its-dependencies)
 - Python3: if not installed, we recommend installing Python3 via [Anaconda](https://www.anaconda.com/products/individual), which is CPython compiled with `clang`
 - `pip` for your Python3 installation: this should be completed if installing via Anaconda above
 - The necessary python packages: via `pip install -r requirements.txt`
 
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
1) Running `pip install banditpam`, OR
2) Running `pip install .` in the home directory (`/BanditPAM`)

## Known Issues 
The following is a list of issues seen when installing BanditPAM on Windows. To report a bug, please file an issue at https://github.com/ThrunGroup/BanditPAM/

None.
