# Changelog

## [0.3.0] - 2020-07-13

### Fixed

- Fix deallocation bug

A bug existed where the memory was deallocated with `free` rather than the deallocator matching the allocator.

### Added

- CI/CD support for Windows, MacOS
- Test support with ctest

### Changed

- Armadillo and Pybind11 as submodules

Armadillo and Pybind11 are no longer shipped with CARMA in the test directory but have been included as submodules.

- Enable use of Armadillo and Pybind11 out of carma repository

This enables CARMA to be used in an existing project with different versions that included as submodules.

- Clang format

All source files have been formatted using clang format.

- Typos

Multiple typos have been correct in the comments and tests.
This change has no influence on API.

## [0.2.0] - 2020-06-28

### Changed 

- Fix spelling of writeable
- Restructure include directory

### Added

- ArrayStore

A class for holding the memory of a Numpy array as an Armadillo matrix in C++ and creating views on the memory as Numpy arrays.
An example use-case would be a C++ class that does not return all data computed, say a Hessian, but should do so on request.
The memory of the views is tied to lifetime of the class.


Functions to edit Numpy flags (OWNDATA, WRITEABLE)
Documentation example on how to take ownership of Numpy array

## [0.1.2] - 2020-05-31

### Added

- Functions to edit Numpy flags (OWNDATA, WRITEABLE)

Functions and documentation example on how to take ownership of Numpy array

## [0.1.2] - 2020-05-22

### Changed

- Fix in CMakelists as interface lib
- Fix non-template type in carma.h
