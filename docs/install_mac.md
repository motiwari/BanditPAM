# Installation Tutorial for MacOS

## Prerequisites
The following is a more detailed description of the installation process of BanditPAM for MacOS. Please ensure the following dependencies are installed:
 - The most recent version of the Xcode Command Line Tools: via `xcode-select --install`
 - Homebrew: via `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
 - LLVM's `clang`: via `brew install llvm`
 - Python3: if not installed, we recommend installing Python3 via [Anaconda](https://www.anaconda.com/products/individual), which is CPython compiled with `clang`
 - `pip` for your Python3 installation; this should be completed if installing via Anaconda above
 - The necessary python packages: via `pip install -r requirements.txt`
 - The OpenMP and and Armadillo libraries: via `brew install libomp armadillo`

(NOT RECOMMENDED): Instead of LLVM's `clang`, you can also use another C++ compiler and point your `CC` environment variable to it. Please ensure this is the same compiler used to compile your Python installation if using CPython. If you open a `python` REPL it will show the compiler used during the language installation:

 ```
 >> python
Python 3.7.9 (default, Mar  1 2021, 13:32:26)
[Clang 11.0.0 (clang-1100.0.33.17)] :: Intel Corporation on darwin
Type "help", "copyright", "credits" or "license" for more information.
Intel(R) Distribution for Python is brought to you by Intel Corporation.
Please check out: https://software.intel.com/en-us/python-distribution
```

For the default Python2 installed on Mac, the `Apple clang` compiler is used; for Intel python, `clang` is used as well. Do not attempt to compile the BanditPAM extension with `gcc` if your Python used `clang`.

## BanditPAM Installation

This should successfully install the requirements needed for BanditPAM, which can then be installed via ONE of the following ways:
1) Running `pip install BanditPAM`, OR
2) Running `pip install .` in the home directory (`/BanditPAM`).

## Known Issues 
The following is a list of issues seen when installing BanditPAM. This is updated as further issues are encountered. To report a bug, please file an issue at https://github.com/ThrunGroup/BanditPAM/

None.
