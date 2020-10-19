import sys
import sysconfig
import os
import tempfile
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

__version__ = '0.0.1'
os.environ["CC"] = "/opt/devtools-6.2/bin/gcc"
os.environ["CXX"] = "/opt/devtools-6.2/bin/g++"


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()


ext_modules = [
    Extension(
        'BanditPAM',
        sorted(['src/kmedoids_ucb.cpp',
                'src/kmeds_pywrapper.cpp']),
        include_dirs=[
            get_pybind_include(),
            'headers',
        ],
        libraries=['armadillo'],
        language='c++14',
        extra_compile_args=['-static-libstdc++'],
    ),
]


def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    # flags = ['-std=c++17', '-std=c++14', '-std=c++11']
    flags = ['-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        opts.append('-Wno-register')
        opts.append('-std=c++14')
        # opts.append('-fopenm')
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            # opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        for ext in self.extensions:
            ext.define_macros = [('VERSION_INFO', '"{}"'.format(self.distribution.get_version()))]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)

setup(
    name='BanditPAM',
    version=__version__,
    author='James Mayclin and Eric Frankel, Mo Tiwari',
    maintainer="Eric Frankel",
    author_email='ericsf@stanford.edu',
    url='https://github.com/jmayclin/BanditPAM',
    description='C++ implementation of BanditPAM algorithm with Python Bindings',
    long_description='This repo contains a high-performance implementation of BanditPAM from https://arxiv.org/abs/2006.06856. The code can be called directly from Python or C++.',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
