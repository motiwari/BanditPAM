#!/bin/bash

# Unzip the files to their appropriate places

cp /content/BanditPAM/scripts/colab_files/usr-include-armadillo /usr/include/armadillo
cp /content/BanditPAM/scripts/colab_files/usr-lib-x86_64-linux-gnu-pkgconfig-armadillo /usr/lib/x86_64-linux-gnu/pkgconfig/armadillo.pc
tar -zxvf /content/BanditPAM/scripts/colab_files/usr-include-armadillo_bits.tar.gz /usr/include
tar -zxvf /content/BanditPAM/scripts/colab_files/usr-lib-x86_64-linux-gnu.tar.gz /usr/lib/x86_64-linux-gnu/
mkdir -p /usr/share/Armadillo
tar -zxvf /content/BanditPAM/scripts/colab_files/usr-share-Armadillo-CMake.tar.gz /usr/share/Armadillo