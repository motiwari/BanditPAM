#!/bin/bash

# Install dependencies on Google Colab machine

apt install -y build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev

# Copies and unzips the files to their appropriate places

cp /content/BanditPAM/scripts/colab_files/usr-include-armadillo /usr/include/armadillo
cp /content/BanditPAM/scripts/colab_files/usr-lib-x86_64-linux-gnu-pkgconfig-armadillo.pc /usr/lib/x86_64-linux-gnu/pkgconfig/armadillo.pc
tar -zxvf /content/BanditPAM/scripts/colab_files/usr-include-armadillo_bits.tar.gz --directory /usr/include
tar -zxvf /content/BanditPAM/scripts/colab_files/usr-lib-x86_64-linux-gnu.tar.gz --directory /usr/lib/x86_64-linux-gnu/
mkdir -p /usr/share/Armadillo
tar -zxvf /content/BanditPAM/scripts/colab_files/usr-share-Armadillo-CMake.tar.gz --directory /usr/share/Armadillo