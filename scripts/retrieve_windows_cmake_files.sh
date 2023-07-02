#!/bin/bash

# This script is used to retrieve the files necessary for the Windows CMake build

echo "Retrieving files for Windows CMake build..."

# Move up one directory
cd ..

# Clone the GitHub repository
git clone https://github.com/ThrunGroup/BanditPAM_Windows

# Move unistd.h to headers/unistd.h
mv BanditPAM_Windows/unistd.h headers/unistd.h
#
# Move getopt.h to headers/getopt.h
mv BanditPAM_Windows/getopt.h headers/getopt.h
#
# Move getopt.cpp to src/getopt.cpp
mv BanditPAM_Windows/getopt.cpp src/getopt.cpp

# Delete BanditPAM_Windows directory
rm -rf BanditPAM_Windows

echo "Done!"