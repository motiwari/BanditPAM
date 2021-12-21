import sys
import os
import tempfile
import setuptools
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import distutils.sysconfig
import distutils.spawn

__version__ = "1.0.5"


class get_pybind_include(object):
    """
    Helper class to determine the pybind11 include path.
    The purpose of this class is to postpone importing pybind11
    until it is actually installed via setup's setup_requires arg,
    so that the ``get_include()`` method can be invoked.
    """
    def __str__(self):
        import pybind11
        return pybind11.get_include()


class get_numpy_include(object):
    """
    Helper class to determine the numpy include path
    The purpose of this class is to postpone importing numpy
    until it is actually installed via setup's setup_requires arg,
    so that the ``get_include()`` method can be invoked.
    """
    def __str__(self):
        import numpy
        return numpy.get_include()


def compiler_check():
    """
    This is necessary because setuptools will use the compiler that compiled
    python -- even if the user specifies another one! -- for some of the
    compilation process
    """
    try:
        return 'clang' if 'clang' in distutils.sysconfig.get_config_vars()['CC'] else 'gcc'
    except KeyError as _e:
        # The 'CC' environment variable hasn't been set. In this case, search for clang and gcc
        # Borrowed from https://github.com/clab/dynet/blob/master/setup.py
        if distutils.spawn.find_executable('clang') is not None:
            return 'clang'
        elif distutils.spawn.find_executable('gcc') is not None:
            return 'gcc'

    raise Exception("No C++ compiler was found. Please install LLVM clang.")


def has_flag(compiler, flagname):
    """
    Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            print("Warning: Received an OSError")
    return True


def cpp_flag(compiler):
    """
    Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    compiler_name = compiler_check()
    if compiler_name == 'clang':
        flags = ["-std=c++17", "-std=c++14", "-std=c++11"]
    else:
        # Assume gcc
        flags = ["-std=c++1y"]

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError("Unsupported compiler -- at least C++11 support is needed!")


def check_brew_package(pkg_name):
    brew_cmd = ["brew", "--prefix", pkg_name]
    process = subprocess.Popen(brew_cmd, stdout=subprocess.PIPE)
    output, _error = process.communicate()
    if output.decode() == "":
        raise Exception(
            "Error: Need to install %s via homebrew! Please run `brew install %s`"
            % (pkg_name, pkg_name)
        )
    return output.decode().strip()


def check_brew_installation():
    brew_cmd = ["which", "brew"]
    process = subprocess.Popen(brew_cmd, stdout=subprocess.PIPE)
    output, _error = process.communicate()
    if output.decode() == "":
        raise Exception("Error: Need to install homebrew! Please see https://brew.sh")


def check_numpy_installation():
    try:
        import numpy
    except ModuleNotFoundError:
        raise Exception("Need to install numpy!")


def install_check_mac():
    # Make sure homebrew is installed
    check_brew_installation()

    # Make sure numpy is installed
    check_numpy_installation()

    # Check that LLVM clang, libomp, and armadillo are installed
    llvm_loc = check_brew_package(
        "llvm"
    )  # We need to use LLVM clang since Apple's clang doesn't support OpenMP
    _libomp_loc = check_brew_package("libomp")
    _arma_loc = check_brew_package("armadillo")

    # Set compiler to LLVM clang on Mac, since Apple clang doesn't support OpenMP
    os.environ["CC"] = os.path.join(llvm_loc, "bin", "clang")


def check_omp_install_linux():
    # TODO: Need to get exact compiler name and version to check this
    # Check compiler version is gcc>=6.0.0 or clang>=X.X.X
    pass


def check_armadillo_install_linux():
    # Since armadillo is a C++ extension, just check if it exists
    cmd = ["find", "/", "-iname", "armadillo"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    output, _error = process.communicate()
    if output.decode() == "":
        print(
            "Warning: Armadillo may not be installed. \
            Please build it from",
            os.path.join(
                "BanditPAM", "headers", "carma", "third_party", "armadillo-code"
            ),
        )
    return output.decode().strip()


def check_linux_package_installation(pkg_name):
    cmd = ["dpkg", "-s", pkg_name]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output, _error = process.communicate()
    if output.decode() == "":
        raise Exception(
            "Error: Need to install %s! \
            Please ensure all dependencies are installed via your package manager (apt, yum, etc.): \
            build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libssl-dev \
            libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev"
            % (pkg_name)
        )
    return output.decode().strip()


def setup_colab():
    # If we're in Google Colab, we need to manually copy over the prebuilt armadillo libraries
    # NOTE: This only works for Google Colab instances with Ubuntu 18.04 Runtimes!
    try:
        import google.colab
        in_colab = True
    except: #TODO: Better exception handling
        in_colab = False
    
    if in_colab:
        # TODO: Dangerous os.system() call: https://stackoverflow.com/a/51329156
        os.system('git clone https://github.com/ThrunGroup/BanditPAM.git /content/BanditPAM')
        os.system('/content/BanditPAM/scripts/colab_install_armadillo.sh')
        os.system('rm -rf /content/BanditPAM')


def install_check_ubuntu():
    # Make sure linux packages are installed
    dependencies = [
        "build-essential",
        "checkinstall",
        "libreadline-gplv2-dev",
        "libncursesw5-dev",
        "libssl-dev",
        "libsqlite3-dev",
        "tk-dev",
        "libgdbm-dev",
        "libc6-dev",
        "libbz2-dev",
        "libffi-dev",
        "zlib1g-dev",
    ]

    setup_colab()

    for dep in dependencies:
        check_linux_package_installation(dep)

    # Make sure numpy is installed
    check_numpy_installation()

    # Check openMP is installed
    check_omp_install_linux()

    # Check armadillo is installed
    check_armadillo_install_linux()


def is_ubuntu():
    cmd = ["cat", "/etc/issue"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output, _error = process.communicate()
    return "Ubuntu" in output.decode()


class BuildExt(build_ext):
    """
    A custom build extension for adding compiler-specific options.
    """

    c_opts = {"msvc": ["/EHsc"], "unix": []}
    l_opts = {"msvc": [], "unix": []}

    if sys.platform == "darwin":
        install_check_mac()
        assert(compiler_check() == 'clang', "Need to install LLVM clang!")
        darwin_opts = ["-stdlib=libc++", "-mmacosx-version-min=10.7", "-O3"]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts
    elif sys.platform == "linux" or sys.platform == "linux2":
        if is_ubuntu():
            install_check_ubuntu()

        linux_opts = ["-O3"]
        c_opts["unix"] += linux_opts
        l_opts["unix"] += linux_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type

        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])

        opts.append("-Wno-register")
        opts.append(cpp_flag(self.compiler))
        opts.append("-O3")  # Add it here as well, in case of Windows installation
        opts.append("-fopenmp")

        # TODO: Change OMP library library name based on gcc vs clang instead of based on OS
        compiler_name = compiler_check()
        if sys.platform == "darwin":
            assert(compiler_name == 'clang', "Need to install LLVM clang!")
            link_opts.append('-lomp')
        else:
            if compiler_name == 'clang':
                link_opts.append("-lomp")
            else:
                link_opts.append("-lgomp")

        if ct == "unix":
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")

        for ext in self.extensions:
            ext.define_macros = [
                ("VERSION_INFO", '"{}"'.format(self.distribution.get_version()))
            ]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts

        build_ext.build_extensions(self)


if sys.platform == "linux" or sys.platform == "linux2":
    include_dirs = [
        get_pybind_include(),
        get_numpy_include(),
        "headers",
        "headers/carma/include/carma_bits",
        "headers/carma/include",
        "/usr/local/include",
    ]
    
else:  # OSX
    include_dirs = [
        get_pybind_include(),
        get_numpy_include(),
        "headers",
        "headers/carma/include",
        "headers/carma/include/carma",
        "headers/carma/include/carma/carma",
        "/usr/local/include",
    ]

compiler_name = compiler_check()
if compiler_name == "clang":
    libraries = ["armadillo", "omp"]
else: # gcc
    libraries = ["armadillo", "gomp"]

ext_modules = [
    Extension(
        "banditpam",
        [
            os.path.join("src", "kmedoids_pywrapper.cpp"),
            os.path.join("src", "kmedoids_algorithm.cpp"),
            os.path.join("src", "pam.cpp"),
            os.path.join("src", "banditpam.cpp"),
            os.path.join("src", "fastpam1.cpp"),
            os.path.join("src", "fit_python.cpp"),
            os.path.join("src", "medoids_python.cpp"),
            os.path.join("src", "build_medoids_python.cpp"),
            os.path.join("src", "labels_python.cpp"),
            os.path.join("src", "steps_python.cpp"),
        ],
        include_dirs=include_dirs,
        library_dirs=["/usr/local/lib"],
        libraries=libraries,
        language="c++1y",  # TODO: modify this based on cpp_flag(compiler)
        extra_compile_args=["-static-libstdc++"],
    )
]

with open(os.path.join("docs", "long_desc.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="banditpam",
    version=__version__,
    author="Mo Tiwari and James Mayclin",
    maintainer="Mo Tiwari",
    author_email="motiwari@stanford.edu",
    url="https://github.com/ThrunGroup/BanditPAM",
    description="BanditPAM: A state-of-the-art, high-performance k-medoids algorithm.",
    long_description=long_description,
    ext_modules=ext_modules,
    setup_requires=["pybind11>=2.5.0", "numpy>=1.18"],
    data_files=[("docs", [os.path.join("docs", "long_desc.rst")])],
    include_package_data=True,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    headers=[
        os.path.join("headers", "kmedoids_algorithm.hpp"),
        os.path.join("headers", "kmedoids_pywrapper.hpp"),
        os.path.join("headers", "banditpam.hpp"),
        os.path.join("headers", "fastpam1.hpp"),
        os.path.join("headers", "pam.hpp"),
    ],
)