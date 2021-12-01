# Installation Tutorial for MacOS

## Prerequisites
The following is a more detailed description of the installation process of BanditPAM for MacOS. Please ensure:
 - The most recent versions of the Xcode Command Line Tools are installed via `xcode-select --install`
 - Homebrew is installed; if not, see https://brew.sh/.
 - Python3 is installed; if not, run `brew install python` (by default, Macs only come with Python2)
 - Pip is installed for your Python3 installation; this should be completed automatically by `brew install python` above.
 - A C++ toolchain is installed, e.g. `gcc` or `clang`. We strongly suggest using LLVM's `clang` via `brew install llvm`.
 - -- You can also use another C++ compiler and ensure your `CC` environment variable `CC` points to it. Please ensure this is the same compiler used to compile your Python installation if using CPython. If you open a python REPL it will show the compiler used during the language installation:

 ```
 >> python
Python 3.7.9 (default, Mar  1 2021, 13:32:26)
[Clang 11.0.0 (clang-1100.0.33.17)] :: Intel Corporation on darwin
Type "help", "copyright", "credits" or "license" for more information.
Intel(R) Distribution for Python is brought to you by Intel Corporation.
Please check out: https://software.intel.com/en-us/python-distribution
```

For the default Python2 installed on Mac, the Apple clang compiler is used; for Intel python, clang is used as well. Do not attempt to compile the BanditPAM extension with `gcc` if your Python used `clang`.

## Additional Packages
First, it's necessary to install the prerequisites for the package. Please do the following:
 - Run `brew install libomp armadillo` to install the OpenMP libraries and enable multithreading in the package
 - Run `pip3 install -r requirements.txt` to install `numpy`, `pandas`, `matplotlib`, and `pybind11`.

This should successfully install the requirements needed for BanditPAM, which can then be installed via ONE of the following ways:
 Choice i) Running `pip3 install BanditPAM`.
 Choice ii) Running `pip3 install .` in the home directory.

## Known Issues 
The following is a list of issues seen when installing BanditPAM. This is updated as further issues are encountered. To report a bug, please file an issue at https://github.com/ThrunGroup/BanditPAM/

None.
