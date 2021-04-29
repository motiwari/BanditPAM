FROM ubuntu:18.04

# Install necessary dependencies
RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        build-essential \
        autoconf \
        automake \
        libtool \
        pkg-config \
        apt-transport-https \
        ca-certificates \
        software-properties-common \
        wget \
        git \
        curl \
        gnupg \
        zlib1g-dev \
        vim \
        g++-7 \
        gdb \
        valgrind \
        locales \
        python3.7-dev \
        libpython3.7-dev \
        python3.7-venv \
        python3-distutils \
        python3-pip \
        locales-all &&\
    apt-get clean

# Install CMake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - &&\
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ bionic main" &&\
    apt-get update &&\
    apt-get install -y cmake

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

RUN /usr/bin/python3.7 -m pip install setuptools wheel
RUN /usr/bin/python3.7 -m pip install numpy cython scipy pytest
RUN /usr/bin/python3.7 -m pip install pytest-valgrind
