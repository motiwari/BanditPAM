# Installation Tutorial for MacOS

The following is a description of the installation process of BanditPAM for MacOS. This assumes that:
 - Homebrew is installed; if not, see https://brew.sh/.
 - Python 3 is installed; if not, see .
 - Homebrew is installed.
 - Some form of C++ tools (i.e. gcc, g++) are installed.

## Installation Procedure
First, it's necessary to install the prerequisites for the package. Please do the following:
 - Run `brew install libomp armadillo`.
 - Run `pip3 install -r requirements.txt` in the main BanditPAM directory; the core library that is needed is `pybind11`.

This should successfully install the requirements needed for BanditPAM, which can then be installed in either of the following ways:
 - Running `pip3 install BanditPAM`.
 - Running `pip3 install .` in the home directory.
 - Running `python3 setup.py install` in the home directory.

## Known Issues 
The following is a list of issues seen when installing BanditPAM. This is updated as further issues are encountered.
