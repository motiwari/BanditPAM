import sys
import sysconfig
import os
import tempfile
import setuptools
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

__version__ = '0.0.30'

class get_pybind_include(object):
    '''
    Helper class to determine the pybind11 include path.
    The purpose of this class is to postpone importing pybind11
    until it is actually installed via setup's setup_requires arg,
    so that the ``get_include()`` method can be invoked.
    '''

    def __str__(self):
        import pybind11
        return pybind11.get_include()

class get_numpy_include(object):
    '''
    Helper class to determine the numpy include path
    The purpose of this class is to postpone importing numpy
    until it is actually installed via setup's setup_requires arg,
    so that the ``get_include()`` method can be invoked.
    '''

    def __str__(self):
        import numpy
        return numpy.get_include()


def has_flag(compiler, flagname):
    '''
    Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    '''
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
    '''
    Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    '''
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']
    #flags = ['-std=c++1y']
    
    # TODO (@Mo): Make sure this works when building for manylinux
    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


def check_brew_package(pkg_name):
    brew_cmd = ['brew', '--prefix', pkg_name]
    process = subprocess.Popen(brew_cmd, stdout=subprocess.PIPE)
    output, _error = process.communicate()
    if output.decode() == '':
        raise Exception('Error: Need to install %s via homebrew! Please run `brew install %s`' % (pkg_name, pkg_name))
    return output.decode().strip()


def check_brew_installation():
    brew_cmd = ['which', 'brew']
    process = subprocess.Popen(brew_cmd, stdout=subprocess.PIPE)
    output, _error = process.communicate()
    if output.decode() == '':
        raise Exception('Error: Need to install homebrew! Please see https://brew.sh')


def install_check_mac():
    # Make sure homebrew is installed
    check_brew_installation()

    # Check that LLVM clang, libomp, and armadillo are installed
    llvm_loc = check_brew_package('llvm') # We need to use LLVM clang since Apple's clang doesn't support OpenMP
    _libomp_loc = check_brew_package('libomp')
    _arma_loc = check_brew_package('armadillo')
    
    # Set compiler to LLVM clang++
    os.environ["CC"] = os.path.join(llvm_loc, 'bin', 'clang')
    
class BuildExt(build_ext):
    '''
    A custom build extension for adding compiler-specific options.
    '''
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }
    
    if sys.platform == 'darwin':
        install_check_mac()
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-O3']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, ['-O3'])
        link_opts = self.l_opts.get(ct, [])
        opts.append('-Wno-register')
        opts.append('-std=c++1y')
        if sys.platform == 'darwin':
            opts.append('-fopenmp')
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

ext_modules = [
    Extension(
        'BanditPAM',
        sorted([os.path.join('src', 'kmedoids_ucb.cpp'),
                os.path.join('src', 'kmeds_pywrapper.cpp')]),
        include_dirs=[
            get_pybind_include(),
            get_numpy_include(),
            'headers',
            'headers/carma/include',
            '/usr/local/include',
        ],
        library_dirs=[
            '/usr/local/lib',
        ],
        libraries=['armadillo', 'omp'],
        language='c++1y',
        extra_compile_args=['-static-libstdc++'],
    ),
]

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
