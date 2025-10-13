# -*- coding: utf-8 -*-
"""Enhanced setup.py for BanditPAM with comprehensive error handling and dependency management."""

import os
import sys
import subprocess
import shutil
import platform
import re
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
from distutils.version import LooseVersion
import pybind11

# Version information
__version__ = "6.0.3"

# System detection
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"
IS_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS", "false").lower() == "true"

def check_system_requirements():
    """Check and report system requirements."""
    requirements = {
        "cmake": {"cmd": "cmake --version", "version_pattern": r"version (\d+\.\d+)"},
        "make": {
            "cmd": "make --version" if not IS_WINDOWS else "nmake /?", 
            "optional": IS_WINDOWS,
        },
        "git": {"cmd": "git --version", "version_pattern": r"version (\d+\.\d+)"},
    }

    missing_requirements = []

    for tool, config in requirements.items():
        try:
            result = subprocess.run(
                config["cmd"].split(), capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                if not config.get("optional", False):
                    missing_requirements.append(tool)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            if not config.get("optional", False):
                missing_requirements.append(tool)

    return missing_requirements

def get_pybind_include():
    """Get pybind11 include directory."""
    try:
        import pybind11
        return pybind11.get_include()
    except ImportError:
        return None

def get_numpy_include():
    """Get numpy include directory."""
    try:
        import numpy as np
        return np.get_include()
    except ImportError:
        return None

def find_armadillo():
    """Find Armadillo installation."""
    search_paths = [
        "/usr/include/armadillo",
        "/usr/local/include/armadillo",
        "/opt/homebrew/include/armadillo",
        "/usr/local/Cellar/armadillo",
    ]

    if IS_WINDOWS:
        search_paths.extend([
            "C:/vcpkg/installed/x64-windows/include/armadillo",
            "C:/Program Files/Armadillo/include",
        ])

    for path in search_paths:
        if os.path.exists(path):
            return os.path.dirname(path)
    return None

class BanditPAMBuildExt(build_ext):
    """Custom build extension for BanditPAM."""

    def build_extensions(self):
        # Check system requirements
        missing_reqs = check_system_requirements()
        if missing_reqs and not IS_GITHUB_ACTIONS:
            print(f"\n❌ Missing system requirements: {', '.join(missing_reqs)}")
            sys.exit(1)

        # Setup compiler flags
        c_opts = {
            "msvc": ["/EHsc", "/std:c++17", "/O2"],
            "unix": ["-std=c++17", "-O3", "-ffast-math"],
        }

        l_opts = {"msvc": [], "unix": []}

        # Platform-specific configurations
        if self.compiler.compiler_type == "msvc":
            c_opts["msvc"].extend(["/DWIN32", "/D_WINDOWS", "/DNOMINMAX"])
        else:
            c_opts["unix"].extend(["-fPIC", "-Wno-unused-function"])
            if IS_MACOS:
                c_opts["unix"].extend(["-stdlib=libc++", "-mmacosx-version-min=10.14"])
                l_opts["unix"].extend(["-stdlib=libc++", "-mmacosx-version-min=10.14"])
            elif IS_LINUX:
                c_opts["unix"].extend(["-fopenmp"])
                l_opts["unix"].extend(["-fopenmp"])

        # Apply flags to all extensions
        ct = self.compiler.compiler_type
        opts = c_opts.get(ct, [])
        link_opts = l_opts.get(ct, [])

        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(link_opts)

            # Add include directories
            pybind_include = get_pybind_include()
            numpy_include = get_numpy_include()
            armadillo_include = find_armadillo()

            if pybind_include:
                ext.include_dirs.append(pybind_include)
            if numpy_include:
                ext.include_dirs.append(numpy_include)
            if armadillo_include:
                ext.include_dirs.append(armadillo_include)
                if not IS_WINDOWS:
                    ext.libraries.extend(["armadillo"])

        super().build_extensions()

def get_extensions():
    """Define extensions to build."""

    # Only files that actually exist
    source_files = [
        "src/algorithms/kmedoids_algorithm.cpp",
        "src/algorithms/pam.cpp", 
        "src/algorithms/banditpam.cpp",
        "src/algorithms/banditpam_orig.cpp",
        "src/algorithms/fastpam1.cpp",
        "src/python_bindings/kmedoids_pywrapper.cpp",
    ]

    # Check if optional files exist and add them
    optional_files = [
        "src/python_bindings/predict_python.cpp",
        "src/python_bindings/sparse_support_python.cpp",
        "src/python_bindings/medoids_python.cpp",
        "src/python_bindings/build_medoids_python.cpp",
        "src/python_bindings/loss_python.cpp",
        "src/python_bindings/cache_python.cpp",
        "src/python_bindings/fit_python.cpp",
        "src/python_bindings/labels_python.cpp",
        "src/python_bindings/loss_fn_python.cpp",
        "src/python_bindings/steps_python.cpp",
        "src/python_bindings/swap_times_python.cpp"
    ]

    for file in optional_files:
        if os.path.exists(file):
            source_files.append(file)

    # Include directories
    include_dirs = [
        "headers/algorithms",
        "headers/python_bindings", 
        "headers/carma/include",
        get_pybind_include(),
        get_numpy_include()
    ]

    # Filter out None values
    include_dirs = [d for d in include_dirs if d is not None]

    # Libraries to link
    libraries = []
    if not IS_WINDOWS:
        libraries.extend(["armadillo"])

    ext = Pybind11Extension(
        "banditpam",
        source_files,
        include_dirs=include_dirs,
        libraries=libraries,
        language='c++',
        cxx_std=17,
    )

    return [ext]

def main():
    """Main setup function."""
    long_description = ""
    readme_path = Path("README.md")
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            long_description = f.read()

    setup(
        name="banditpam",
        version=__version__,
        author="Mo Tiwari",
        author_email="motiwari@stanford.edu",
        description="BanditPAM: Almost Linear-Time k-Medoids Clustering",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/motiwari/BanditPAM",
        packages=find_packages(),
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BanditPAMBuildExt},
        python_requires=">=3.8",
        setup_requires=[
            "setuptools>=45.0.0",
            "wheel",
            "pybind11>=2.10.0",
            "numpy>=1.18.0",
        ],
        install_requires=[
            "numpy>=1.18.0",
            "pybind11>=2.10.0",
        ],
        extras_require={
            "plotting": ["matplotlib>=3.0.0"],
            "examples": ["pandas>=1.0.0", "scikit-learn>=0.24.0", "matplotlib>=3.0.0"],
            "dev": ["pytest>=6.0.0", "black", "flake8", "mypy"],
            "all": ["matplotlib>=3.0.0", "pandas>=1.0.0", "scikit-learn>=0.24.0"],
        },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers", 
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9", 
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: C++",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        keywords=["clustering", "k-medoids", "machine-learning", "bandit-algorithms"],
        license="MIT",
        zip_safe=False,
        include_package_data=True,
    )

if __name__ == "__main__":
    main()
