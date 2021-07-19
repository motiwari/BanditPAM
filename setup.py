import sys
import sysconfig
import os
import tempfile
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

__version__ = '0.0.21'
#os.environ["CC"] = os.path.join('/usr', 'bin', 'gcc')
#os.environ["CXX"] = os.path.join('/usr', 'bin', 'g++')


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed via setup's setup_requires arg,
    so that the ``get_include()`` method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()

class get_numpy_include(object):
    """Helper class to determine the numpy include path
    The purpose of this class is to postpone importing numpy
    until it is actually installed via setup's setup_requires arg,
    so that the ``get_include()`` method can be invoked. """

    def __str__(self):
        import numpy
        return numpy.get_include()

ext_modules = [
    Extension(
        'BanditPAM',
        sorted([os.path.join('src', 'kmedoids_ucb.cpp'),
                os.path.join('src', 'kmedoids_pywrapper.cpp')]),
        include_dirs=[
            get_pybind_include(),
            get_numpy_include(),
            'headers',
            'headers/carma/include',
            'headers/carma/include/carma',
            'headers/carma/include/carma/carma',
            
        ],
        libraries=['armadillo'],
        language='c++1y',
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
    flags = ['-std=c++1y']

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
        link_opts = self.l_opts.get(ct, [])
        opts.append('-Wno-register')
        opts.append('-std=c++1y')
        if sys.platform == 'darwin':
            opts.append('-Xpreprocessor -fopenmp')
        else:
            opts.append('-fopenmp')
            link_opts.append('-lgomp')
        if ct == 'unix':
            # opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        for ext in self.extensions:
            ext.define_macros = [('VERSION_INFO', '"{}"'.format(self.distribution.get_version()))]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


with open(os.path.join('docs', 'long_desc.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='BanditPAM',
    version=__version__,
    author='Mo Tiwari and James Mayclin',
    maintainer="Mo Tiwari",
    author_email='motiwari@stanford.edu',
    url='https://github.com/ThrunGroup/BanditPAM',
    description='BanditPAM: A state-of-the-art, high-performance k-medoids algorithm.',
    long_description=long_description,
    ext_modules=ext_modules,
    setup_requires=[
        'pybind11>=2.5.0',
        'numpy>=1.18',
    ],
    data_files=[('docs', [os.path.join('docs', 'long_desc.rst')])],
    include_package_data=True,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    headers=[os.path.join('headers', 'kmedoids_ucb.hpp')],
)
