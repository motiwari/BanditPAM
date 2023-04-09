# Installation Tutorial for Windows

The following is a description of the installation process of BanditPAM for Windows. This assumes that:
 
## Prerequisites
Please ensure the following dependencies are installed:
 - Visual Studio (2022 is recommended) with "Desktop development with C++" workload: via the [Visual Studio download page](https://visualstudio.microsoft.com/vs/)
 - `CMake`: via the [CMake installation instructions](https://cmake.org/install/)
 - `git`: via the [Git installation instructions](https://git-scm.com/download/win)
 - Python3: if not installed, we recommend installing Python3 via [Anaconda](https://www.anaconda.com/products/individual), which is CPython compiled with `clang`
 - `pip` for your Python3 installation: this should be completed if installing via Anaconda above

## BanditPAM Installation

### C++ Build

Open a command prompt and run the following commands:
1) `git clone https://github.com/motiwari/BanditPAM.git`
2) `cd BanditPAM`
3) `git submodule update --init --recursive`
4) `pip install -r requirements.txt`
5) `cd headers\carma`
6) `git checkout stable`
7) `cd ..`
8) `git clone https://gitlab.com/conradsnicta/armadillo-code.git`
9) `cd ..`

Edit the following files:
1) Replace `src/CMakeLists.txt` with the following:
```text
include_directories(${PROJECT_SOURCE_DIR}/headers)
include_directories(${PROJECT_SOURCE_DIR}/headers/algorithms)
include_directories(${PROJECT_SOURCE_DIR}/headers/python_bindings)
include_directories(${PROJECT_SOURCE_DIR}/headers/carma/include)
include_directories(${PROJECT_SOURCE_DIR}/headers/carma/include/carma_bits)
include_directories(${PROJECT_SOURCE_DIR}/headers/armadillo-code/include)
include_directories(${PROJECT_SOURCE_DIR}/headers/armadillo-code/include/armadillo_bits)

# For Mac Github Runner to find OpenMP -- potentially unnecessary after fixing -Xpreprocessor -fopenmp issue
include_directories(/usr/local/Cellar/libomp/15.0.2/include)
include_directories(/usr/local/Cellar/libomp/15.0.7/include)
include_directories(/usr/local/Cellar/libomp)
include_directories(/usr/local/opt/libomp/lib)
include_directories(/usr/local/opt/libomp)
include_directories(/usr/local/opt)

set(OPENBLAS_dir ${PROJECT_SOURCE_DIR}/headers/armadillo-code/examples/lib_win64)
add_executable(BanditPAM getopt.cpp main.cpp )

add_library(BanditPAM_LIB algorithms/kmedoids_algorithm.cpp algorithms/pam.cpp algorithms/banditpam.cpp algorithms/banditpam_orig.cpp algorithms/fastpam1.cpp)
target_link_directories(BanditPAM PUBLIC ${OPENBLAS_dir})
target_link_libraries(BanditPAM_LIB libopenblas)
target_link_libraries(BanditPAM PUBLIC BanditPAM_LIB)

string(REPLACE "/" "\\"  OPENBLAS_dir_win ${OPENBLAS_dir})
message(STATUS ${OPENBLAS_dir})
message(STATUS ${OPENBLAS_dir_win}\\dll)
add_custom_command(TARGET BanditPAM_LIB POST_BUILD
    COMMAND copy ${OPENBLAS_dir_win}\\*.dll $(OutDir)
)
```
2) Add the `from pybind11.setup_helpers import Pybind11Extension, build_ext` import at the top of `setup.py` and replace from `class BuildExt(build_ext):` to the bottom of the file with the following:
```python
class BuildExt(build_ext):
    """
    A custom build extension for adding compiler-specific options.
    """
    def build_extensions(self):
        for ext in self.extensions:
            ext.define_macros = [
                ("VERSION_INFO", '"{}"'.format(
                    self.distribution.get_version()
                ))
            ]
        build_ext.build_extensions(self)

include_dirs = [
    get_pybind_include(),
    get_numpy_include(),
    "headers",
    os.path.join("headers", "algorithms"),
    os.path.join("headers", "python_bindings"),
    os.path.join("headers", "carma", "include"),
    os.path.join("headers", "carma", "include", "carma_bits"),
    os.path.join("headers", "armadillo-code", "include"),
    os.path.join("headers", "armadillo-code", "include", "armadillo_bits"),
]

cpp_args = ['/std:c++17']
ext_modules = [
    Pybind11Extension(
        "banditpam",
        [
            os.path.join("src", "algorithms", "kmedoids_algorithm.cpp"),
            os.path.join("src", "algorithms", "pam.cpp"),
            os.path.join("src", "algorithms", "banditpam.cpp"),
            os.path.join("src", "algorithms", "banditpam_orig.cpp"),
            os.path.join("src", "algorithms", "fastpam1.cpp"),
            os.path.join("src", "python_bindings",
                            "kmedoids_pywrapper.cpp"),
            os.path.join("src", "python_bindings", "medoids_python.cpp"),
            os.path.join("src", "python_bindings",
                            "build_medoids_python.cpp"),
            os.path.join("src", "python_bindings", "fit_python.cpp"),
            os.path.join("src", "python_bindings", "labels_python.cpp"),
            os.path.join("src", "python_bindings", "steps_python.cpp"),
            os.path.join("src", "python_bindings", "loss_python.cpp"),
            os.path.join("src", "python_bindings", "cache_python.cpp"),
            os.path.join("src", "python_bindings",
                            "swap_times_python.cpp"),
        ],
        include_dirs=include_dirs,
        # language="c++",  # TODO: modify this based on cpp_flag(compiler)
        extra_compile_args=cpp_args,
        libraries = ['libopenblas'],
        library_dirs = [os.path.join(os.getcwd(),r'headers\armadillo-code\examples\lib_win64')],
        # define_macros = [('VERSION_INFO', __version__)],
    )
]

with open(os.path.join("docs", "long_desc.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="banditpam",
    version=__version__,
    author="Mo Tiwari",
    maintainer="Mo Tiwari",
    author_email="motiwari@stanford.edu",
    url="https://github.com/ThrunGroup/BanditPAM",
    description="BanditPAM: A state-of-the-art, \
        high-performance k-medoids algorithm.",
    long_description=long_description,
    ext_modules=ext_modules,
    setup_requires=["pybind11>=2.5.0", "numpy>=1.18"],
    data_files=[("docs", [os.path.join("docs", "long_desc.rst")]),
                ('', [os.path.join(os.getcwd(),r'headers\armadillo-code\examples\lib_win64\libopenblas.dll')])],
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.7",
    cmdclass={"build_ext": BuildExt},
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ],
    # headers=[
    #     os.path.join("headers", "algorithms", "kmedoids_algorithm.hpp"),
    #     os.path.join("headers", "algorithms", "banditpam.hpp"),
    #     os.path.join("headers", "algorithms", "fastpam1.hpp"),
    #     os.path.join("headers", "algorithms", "pam.hpp"),
    #     os.path.join("headers", "python_bindings",
    #                  "kmedoids_pywrapper.hpp"),
    # ],
    # packages=find_packages(),
)

```
3) Add `#include <getopt.h>` to `src/main.cpp` and comment out `printf("unknown option: %c\n", optopt);`

Add the following files:
4) `headers/algorithms/unistd.h`
5) `headers/getopt.h`
6) `src/getopt.cpp`

Compile the `.exe` file:
1) Add the Visual Studio IDE location to PATH in Environment Variables (e.g. `C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE`). You can find this path by running `C:\Program Files (x86)\Microsoft Visual Studio\Installer>vswhere.exe`.
2) Add `${project_dir}\headers\armadillo-code\examples\lib_win64` to PATH 
3) In a command prompt, do the following:
   1) Change directory to the home directory (`/BanditPAM`)
   2) Run `mkdir build`
   3) `cd build`
   4) `cmake ..`
   5) `devenv BanditPAM.sln /Build "Release|x64"`
4) The `.exe` will be located in `build/src/Release`

### Python Build

BanditPAM can be installed via running `pip install .` in the home directory (`/BanditPAM`).