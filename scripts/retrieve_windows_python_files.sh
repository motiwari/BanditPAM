#!/bin/bash

# This script is used to retrieve the files necessary for the Windows Python build
echo "Retrieving files for Windows Python build..."
cd ..
git clone https://github.com/ThrunGroup/BanditPAM_Windows
cd build
cd -- "$(find . -name 'lib.*')"
mv ../../BanditPAM_Windows/clang_rt.asan_dynamic-x86_64.dll clang_rt.asan_dynamic-x86_64.dll
rm -rf ../../BanditPAM_Windows
echo "Done!"