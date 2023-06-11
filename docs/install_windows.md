# Installation Tutorial for Windows

The following is a more detailed description of the installation process of BanditPAM for Windows.
 
## Prerequisites
Please ensure the following dependencies are installed:
 - Visual Studio (2022 is recommended) with "Desktop development with C++" workload: via the [Visual Studio download page](https://visualstudio.microsoft.com/vs/)
 - The Armadillo library: via `git clone https://gitlab.com/conradsnicta/armadillo-code.git armadillo` in the `headers` directory
 - CARMA: via the instructions in [its guide](https://github.com/RUrlus/carma#installation)
 - Python3: if not installed, we recommend installing Python3 via [Anaconda](https://www.anaconda.com/products/individual), which is CPython compiled with `clang`
 - `pip` for your Python3 installation: this should be completed if installing via Anaconda above
 - The necessary python packages: via `pip install -r requirements.txt`

## BanditPAM Installation

Copy the files in [this repo](https://github.com/ThrunGroup/BanditPAM-Windows) into the following locations:
1) `headers/unistd.h`
2) `headers/getopt.h`
3) `src/getopt.cpp`

CMake Build:
1) Add the Visual Studio IDE location to PATH in Environment Variables (e.g. `C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE`). You can find this path by running `C:\Program Files (x86)\Microsoft Visual Studio\Installer>vswhere.exe`.
2) Add `${project_dir}\headers\armadillo\examples\lib_win64` to PATH 
3) Run `devenv BanditPAM.sln /Build "Release|x64"` after `cmake ..`
4) The `.exe` will be located in `build/src/Release`

BanditPAM can then be installed via one of the following ways:
1) Running `pip install banditpam`, OR
2) Running `pip install .` in the home directory (`/BanditPAM`)