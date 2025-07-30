#!/bin/bash

# This script is used to retrieve the files necessary for the Windows CMake build
echo "Retrieving files for Windows CMake build..."
cd ..
git clone https://github.com/ThrunGroup/BanditPAM_Windows
mv BanditPAM_Windows/unistd.h headers/unistd.h
mv BanditPAM_Windows/getopt.h headers/getopt.h
mv BanditPAM_Windows/getopt.cpp src/getopt.cpp
rm -rf BanditPAM_Windows
echo "Done!"