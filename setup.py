import sys
import os
import tempfile
import setuptools
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import distutils.sysconfig
import distutils.spawn

__version__ = "6.0.2"

# TODO(@motiwari): Move this to a separate file
GHA = "GITHUB_ACTIONS"


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
    python for some of the compilation process, even if the user specifies
    another one!
    """
    if sys.platform == "darwin":
        # Note: On Mac, we should always use LLVM clang
        # because Apple's clang does not support OpenMP
        # and we need OpenMP to parallelize the code
        # NOTE: distutils.sysconfig.get_config_vars() works for Python 3.11+,
        #  but we need to use os.environ for Python 3.10
        if "clang" not in distutils.sysconfig.get_config_vars()["CC"] and \
                "clang" not in dict(os.environ)["CC"]:
            raise Exception(
                "Need to install LLVM clang! \
                Please run `brew install llvm`"
            )
        return "clang"

    # For all platforms
    try:
        return (
            "clang"
            if "clang" in distutils.sysconfig.get_config_vars()["CC"]
            else "gcc"
        )
    except KeyError:
        # The 'CC' environment variable hasn't been set.
        # In this case, search for compilers that we can use.
        # Borrowed from https://github.com/clab/dynet/blob/master/setup.py
        if distutils.spawn.find_executable("cl") is not None:
            return "msvc"
        elif distutils.spawn.find_executable("clang") is not None:
            return "clang"
        elif distutils.spawn.find_executable("gcc") is not None:
            return "gcc"

    raise Exception(
        "No C++ compiler was found. Please ensure you have "
        "MSVC, LLVM clang, or GCC."
    )


def has_flag(compiler: str, flagname: str):
    """
    Return a boolean indicating whether a flag name is supported on the
    specified compiler.
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


def cpp_flag(compiler: str):
    """
    Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    compiler_name = compiler_check()
    if compiler_name == "clang" or compiler_name == "gcc":
        flags = ["-std=c++17"]
    else:
        # Assume msvc
        flags = ["/std:c++17"]  # required for std::optional

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError(
        "Unsupported compiler -- \
        at least C++11 support is needed!"
    )


def check_brew_package(pkg_name: str):
    brew_cmd = ["brew", "--prefix", pkg_name]
    process = subprocess.Popen(brew_cmd, stdout=subprocess.PIPE)
    output, _error = process.communicate()
    if output.decode() == "":
        raise Exception(
            "Error: Need to install %s via homebrew! \
            Please run `brew install %s`"
            % (pkg_name, pkg_name)
        )

    return output.decode().strip()


def check_brew_installation():
    brew_cmd = ["which", "brew"]
    process = subprocess.Popen(brew_cmd, stdout=subprocess.PIPE)
    output, _error = process.communicate()
    if output.decode() == "":
        raise Exception(
            "Error: Need to install homebrew! \
            Please see https://brew.sh"
        )


def check_numpy_installation():
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        raise Exception("Need to install numpy!")


def install_check_mac():
    # Make sure homebrew is installed
    check_brew_installation()

    # Make sure numpy is installed
    check_numpy_installation()

    # Check that libomp, armadillo are installed
    check_brew_package("libomp")
    check_brew_package("armadillo")
    check_brew_package("openblas")

    # Because we don't want to force M1 users to compile from source,
    # we build wheels on Github Runners via Github Actions. In
    # doing so, we need to use Apple's compiler to cross-compile
    # for M1 Macs and should NOT use clang.
    # If we are building from source on a non-M1 Mac, we should use
    # LLVM clang to support multithreading via OpenMP
    # TODO(@motiwari): Check if the arm64 wheels are not multithreaded
    # TODO(@motiwari): Check if the universal2 wheels are not multithreaded
    #  when installed on Intel Macs
    if not os.environ.get(GHA, False):
        # If we are NOT running inside a Github action,
        # check that LLVM clang is installed
        llvm_loc = check_brew_package(
            "llvm"
        )  # We need to use LLVM clang since Apple's doesn't support OpenMP

        # Set compiler to LLVM clang on Mac for OpenMP support
        distutils.sysconfig.get_config_vars()["CC"] = os.path.join(
            llvm_loc, "bin", "clang"
        )


def check_omp_install_linux():
    # TODO: Need to get exact compiler name and version to check this
    # Check compiler version is gcc>=6.0.0 or clang>=X.X.X
    pass


def check_armadillo_install_linux():
    # Since armadillo is a C++ extension, just check if it exists
    cmd = ["find", "/", "-iname", "armadillo"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    output, _error = process.communicate()
    if output.decode() == "":
        print(
            "Warning: Armadillo may not be installed. \
            Please build it from",
            os.path.join(
                "BanditPAM",
                "headers",
                "carma",
                "third_party",
                "armadillo-code",
            ),
        )

    return output.decode().strip()


def check_linux_package_installation(pkg_name: str):
    cmd = ["dpkg", "-s", pkg_name]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output, _error = process.communicate()
    if output.decode() == "":
        raise Exception(
            "Error: Need to install %s! \
            Please ensure all dependencies are installed \
            via your package manager (apt, yum, etc.): \
            build-essential checkinstall \
            libssl-dev libsqlite3-dev tk-dev \
            libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev"
            % (pkg_name)
        )

    return output.decode().strip()


def install_ubuntu_pkgs():
    # TODO: Remove dangerous os.system() calls
    # See https://stackoverflow.com/a/51329156
    os.system("apt update")
    os.system(
        "apt install -y \
        build-essential \
        checkinstall \
        libssl-dev \
        libsqlite3-dev \
        libgdbm-dev \
        libc6-dev \
        libbz2-dev \
        libffi-dev \
        zlib1g-dev"
    )  # noqa: E124


def setup_colab(delete_source=False):
    # If we're in Google Colab, we need to manually copy over
    # the prebuilt armadillo libraries
    # NOTE: This only works for Colab instances with Ubuntu 18.04 Runtimes!
    try:
        import google.colab  # noqa: F401

        in_colab = True
    except ModuleNotFoundError:
        in_colab = False

    if in_colab:
        # TODO: Remove dangerous os.system() calls
        # See https://stackoverflow.com/a/51329156
        install_ubuntu_pkgs()

        # TODO(@motiwari): Make this a randomly-named directory
        # and set delete_source=True always
        repo_location = os.path.join("/", "content", "BanditPAM")
        # Note the space after the git URL to separate the source and target
        os.system(
            "git clone https://github.com/motiwari/BanditPAM.git "
            + repo_location
        )
        os.system(
            repo_location
            + os.path.join(
                "scripts",
                "colab_files",
                "colab_install_armadillo.sh",
            )
        )
        if delete_source:
            os.system("rm -rf " + repo_location)


def setup_paperspace_gradient():
    # Unfortunately, Paperspace Gradient does not make it easy to tell
    # whether you're inside a Gradient instance. For this reason, we
    # determine whether python is running inside a Gradient instance
    # by checking for a /notebooks directory

    # TODO: Remove dangerous os.system() calls
    # See https://stackoverflow.com/a/51329156

    in_paperspace_gradient = os.path.exists("/notebooks")

    if in_paperspace_gradient:
        install_ubuntu_pkgs()
        os.system(
            "DEBIAN_FRONTEND=noninteractive TZ=America/NewYork \
            apt install -y tk-dev"
        )
        os.system(
            "git clone \
            https://gitlab.com/conradsnicta/armadillo-code.git"
        )
        os.system("cd armadillo-code && cmake . && make install")


def install_check_ubuntu():
    # Make sure linux packages are installed
    dependencies = [
        "build-essential",
        "checkinstall",
        "libssl-dev",
        "libsqlite3-dev",
        "tk-dev",
        "libgdbm-dev",
        "libc6-dev",
        "libbz2-dev",
        "libffi-dev",
        "zlib1g-dev",
    ]

    setup_colab(delete_source=False)

    setup_paperspace_gradient()

    for dep in dependencies:
        check_linux_package_installation(dep)

    # Make sure numpy is installed
    check_numpy_installation()

    # Check openMP is installed
    check_omp_install_linux()

    # Check armadillo is installed
    check_armadillo_install_linux()


def is_ubuntu():
    cmd = ["cat", os.path.join("/", "etc", "issue")]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output, _error = process.communicate()
    return "Ubuntu" in output.decode()


def get_package_prefix(package=None):
    assert package is not None, "Package name must be provided"
    try:
        result = subprocess.run(
            ["brew", "--prefix", package],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return result.stdout.strip().decode()
    except subprocess.CalledProcessError as e:
        print("Error running brew:", e.stderr)
        return None


class BuildExt(build_ext):
    """
    A custom build extension for adding compiler-specific options.
    """

    c_opts = {"msvc": ["/EHsc"], "unix": []}
    l_opts = {"msvc": [], "unix": []}

    if sys.platform == "darwin":
        install_check_mac()
        # Verify that we're either compiling with clang or
        # inside a Github Action
        assert compiler_check() == "clang" or os.environ.get(
            GHA, False
        ), "Need to install LLVM clang!"
        darwin_opts = [
            "-stdlib=libc++",
            "-mmacosx-version-min=10.14",
            "-O3",
        ]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts
    elif sys.platform == "linux" or sys.platform == "linux2":
        if is_ubuntu():
            install_check_ubuntu()

        linux_opts = ["-O3"]
        c_opts["unix"] += linux_opts
        l_opts["unix"] += linux_opts
    # Currently necessary (unsure why)
    elif sys.platform == "win32":
        c_opts["msvc"] += ["/fsanitize=address"]

    def build_extensions(self):  # noqa: C901
        ct = self.compiler.compiler_type

        comp_opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])

        # TODO(@motiwari): on Windows, these flags are unrecognized
        comp_opts.append(cpp_flag(self.compiler))
        comp_opts.append("-O3")

        if sys.platform == "darwin":
            for package in ["armadillo"]:  # "libomp"
                package_prefix = get_package_prefix(package)
                link_opts.append(f"-Wl,-rpath,{package_prefix}/lib")

            if os.environ.get(GHA, False):
                # We are inside a Github Runner on a Mac.

                # NOTE: The next two lines NEED TO BE TOGETHER
                comp_opts.append("-Xpreprocessor")
                comp_opts.append("-fopenmp")

                # comp_opts.append("-lomp")  # Potentially unused?

                # TODO(@motiwari): include armadillo here?

                libomp_prefix = get_package_prefix("libomp")
                libomp_static = os.path.join(libomp_prefix, "lib", "libomp.a")

                comp_opts += [f"-I{os.path.join(libomp_prefix, 'include')}"]
                link_opts += [libomp_static, "-lm"]
                link_opts.append("-Wl,-force_load," + libomp_static)

                link_opts += ["-lm"]
                for package in ["libomp"]:
                    package_prefix = get_package_prefix(package)
                    comp_opts.append("-I{}/include".format(package_prefix))
                    comp_opts.append("-L{}/lib".format(package_prefix))  # Remove?
                    link_opts.append("-L{}/lib".format(package_prefix))
        elif sys.platform == "linux":
            link_opts += ["-lm", "-lpthread"]
        elif sys.platform != "win32":
            comp_opts.append("-fopenmp")

        compiler_name = compiler_check()
        # if sys.platform == "darwin":
            # if compiler_name == "gcc":
            #      link_opts.append("-lomp")
            # else:
            #     link_opts.append("-lomp")
        if sys.platform == "linux" or sys.platform == "linux2":
            if compiler_name == "gcc":
                link_opts += ["-lgomp", "-lm", "-lpthread"]
            else:
                # Both clang and Apple's clang should use libomp
                link_opts += ["-lomp", "-lm", "-lpthread"]
        else:
            # On Windows
            # TODO(@motiwari): Fix this
            pass

        if ct == "unix":
            if has_flag(self.compiler, "-fvisibility=hidden"):
                comp_opts.append("-fvisibility=hidden")

        for ext in self.extensions:
            ext.define_macros = [
                (
                    "VERSION_INFO",
                    '"{}"'.format(self.distribution.get_version()),
                )
            ]
            ext.extra_compile_args = comp_opts
            ext.extra_compile_args += []  # []["-arch", "x86_64"]

            ext.extra_link_args = link_opts
            ext.extra_link_args += [
                "-v",
            ] # "-arch", "x86_64"]
            ext.extra_link_args =  link_opts

        build_ext.build_extensions(self)


def main():
    if sys.platform == "linux" or sys.platform == "linux2":
        include_dirs = [
            get_pybind_include(),
            get_numpy_include(),
            "headers",
            os.path.join("headers", "algorithms"),
            os.path.join("headers", "python_bindings"),
            os.path.join("../", "carma", "include"),
            # os.path.join("headers", "carma", "include"),
            # os.path.join("headers", "carma", "include", "carma_bits"),
            os.path.join("/", "usr", "local", "include"),
        
        ]
    elif sys.platform == "darwin":  # OSX
        include_dirs = [
            get_pybind_include(),
            get_numpy_include(),
            "headers",
            os.path.join("headers", "algorithms"),
            os.path.join("headers", "python_bindings"),
            os.path.join("headers", "carma", "include"),
            os.path.join("headers", "carma", "include", "carma"),
            os.path.join("headers", "carma", "include", "carma", "carma"),
            # To include carma when the BanditPAM repo hasnt been initialized
            os.path.join("/", "usr", "local", "include", "carma"),
            os.path.join(
                "/", "usr", "local", "include", "carma", "carma_bits"
            ),
            # When building from source on M1 Macs, may need these dirs
            # Currently, we should never be building from source on an M1 Mac,
            # Only cross-compiling from an Intel Mac
            # TODO(@motiwari): Remove extraneous directories
            os.path.join("/", "opt", "homebrew", "opt", "armadillo"),
            os.path.join("/", "opt", "homebrew", "opt", "openblas"),
            os.path.join(
                "/", "opt", "homebrew", "opt", "armadillo", "include"
            ),
            os.path.join(
                "/",
                "opt",
                "homebrew",
                "opt",
                "armadillo",
                "include",
                "armadillo_bits",
            ),
        ]
        for package in ["armadillo", "libomp"]:
            package_prefix = get_package_prefix(package)
            include_dirs.append(os.path.join(package_prefix, "include"))
    elif sys.platform == "win32":  # WIN32
        include_dirs = [
            get_pybind_include(),
            get_numpy_include(),
            "headers",
            os.path.join("headers", "algorithms"),
            os.path.join("headers", "python_bindings"),
            os.path.join("headers", "carma", "include"),
            os.path.join("headers", "carma", "include", "carma_bits"),
            os.path.join("headers", "armadillo", "include"),
            os.path.join("headers", "armadillo", "include", "armadillo_bits"),
        ]
    else:
        raise Exception("Unrecognized platform")

    compiler_name = compiler_check()
    if sys.platform == "darwin" and os.environ.get(GHA, False):
        # Do NOT link omp here because it'll add an -lomp flag to dynamically link it
        #  (and we want to statically link it)
        libraries = ["armadillo"]
    elif sys.platform == "win32":
        libraries = ["libopenblas"]
    else:
        if compiler_name == "gcc":
            # TODO(@motiwari): Change to c++ when we update cpp_args below?
            libraries = ["armadillo", "gomp", "stdc++"]
        else:  # clang
            libraries = ["armadillo", "omp", "stdc++"]

    if sys.platform == "win32":
        cpp_args = ["/std:c++17"]
        library_dirs = [
            # for windows
            os.path.join(os.getcwd(), r"headers\armadillo\examples\lib_win64"),
        ]
    else:
        cpp_args = [
            "-static-libstdc++"
        ]  # TODO(@motiwari): Modify this based on GCC or Clang
        library_dirs = [
            os.path.join("/", "usr", "local", "lib"),
        ]

    ext_modules = [
        Extension(
            "banditpam",
            [
                os.path.join("src", "algorithms", "kmedoids_algorithm.cpp"),
                os.path.join("src", "algorithms", "pam.cpp"),
                os.path.join("src", "algorithms", "banditpam.cpp"),
                os.path.join("src", "algorithms", "banditpam_orig.cpp"),
                os.path.join("src", "algorithms", "fastpam1.cpp"),
                os.path.join(
                    "src", "python_bindings", "kmedoids_pywrapper.cpp"
                ),
                os.path.join("src", "python_bindings", "medoids_python.cpp"),
                os.path.join(
                    "src", "python_bindings", "build_medoids_python.cpp"
                ),
                os.path.join("src", "python_bindings", "fit_python.cpp"),
                os.path.join("src", "python_bindings", "labels_python.cpp"),
                os.path.join("src", "python_bindings", "steps_python.cpp"),
                os.path.join("src", "python_bindings", "loss_python.cpp"),
                os.path.join("src", "python_bindings", "cache_python.cpp"),
                os.path.join(
                    "src", "python_bindings", "swap_times_python.cpp"
                ),
                
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            language="c++1z",  # TODO: modify this based on cpp_flag(compiler)
            extra_compile_args=cpp_args,
            # Not passed? (BuildExt sets extra_link_args)
            extra_link_args=["-vvvv"],
        )
    ]

    with open(os.path.join("docs", "long_desc.rst"), encoding="utf-8") as f:
        long_description = f.read()

    my_data_files = [("docs", [os.path.join("docs", "long_desc.rst")])]
    if sys.platform == "win32":
        my_data_files.append(
            (
                "",
                [
                    os.path.join(
                        os.getcwd(),
                        r"headers\armadillo\examples\lib_win64"
                        + r"\libopenblas.dll",
                    )
                ],
            )
        )

    setup(
        name="banditpam",
        version=__version__,
        author="Mo Tiwari",
        maintainer="Mo Tiwari",
        author_email="motiwari@stanford.edu",
        url="https://github.com/motiwari/BanditPAM",
        long_description=long_description,
        ext_modules=ext_modules,
        setup_requires=["pybind11>=2.5.0", "numpy>=1.18"],
        data_files=my_data_files,
        include_package_data=True,
        cmdclass={"build_ext": BuildExt},
        zip_safe=False,
        headers=[
            os.path.join("headers", "algorithms", "kmedoids_algorithm.hpp"),
            os.path.join("headers", "algorithms", "banditpam.hpp"),
            os.path.join("headers", "algorithms", "fastpam1.hpp"),
            os.path.join("headers", "algorithms", "pam.hpp"),
            os.path.join(
                "headers", "python_bindings", "kmedoids_pywrapper.hpp"
            ),
        ],
    )


if __name__ == "__main__":
    main()
