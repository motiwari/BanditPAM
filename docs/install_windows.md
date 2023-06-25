# Installation Tutorial for Windows

The following is a more detailed description of the installation process of BanditPAM for Windows.
 
## Prerequisites
Please ensure the following dependencies are installed:
 - Visual Studio (2022 is recommended) with "Desktop development with C++" workload: via the [Visual Studio download page](https://visualstudio.microsoft.com/vs/)
 - The Armadillo library: via `git clone https://gitlab.com/conradsnicta/armadillo-code.git armadillo` in the `headers` directory
 - CARMA: via the instructions in [its guide](https://github.com/RUrlus/carma#installation)
 - Python3: if not installed, we recommend installing Python3 via [Anaconda](https://www.anaconda.com/products/individual)
 - `pip` for your Python3 installation: this should be completed if installing via Anaconda above
 - The necessary python packages: via `pip install -r requirements.txt`

## BanditPAM Installation

CMake Build:

1) Run `scripts/retrieve_windows_cmake_files.sh` to retrieve the files necessary for the Windows CMake build
2) Add the Visual Studio IDE location to PATH in Environment Variables (e.g. `C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE`).
3) Add `${project_dir}\headers\armadillo\examples\lib_win64` to PATH 
4) Run `devenv BanditPAM.sln /Build "Release|x64"` after `cmake ..` in a prompt other than Git Bash
5) The `.exe` will be located in `build/src/Release`

Python Build:
1) Run `python -m pip install banditpam`, OR
2) Follow these steps:
   1) Add the location of `cl.exe` to PATH in Environment Variables (e.g. `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\bin\Hostx64\x64`).
   2) Run `python -m pip install .` in the home directory (`/BanditPAM`)
   2) Add the file `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\bin\Hostx86\x64\clang_rt.asan_dynamic-x86_64.dll` to `build\lib.win-amd64-cpython-310`
   3) Run `python -m pip install .` in the home directory (`/BanditPAM`)