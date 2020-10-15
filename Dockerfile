FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
        build-essential \
        git \
        vim \
        emacs \
        parallel \
        ca-certificates \
        libjpeg-dev \
        wget \
        libopenblas-dev \
        liblapack-dev \
        libarpack2-dev \
        libsuperlu-dev \
        libomp-dev \
        libssl-dev \
        hdf5-tools && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir bandits \
    && cd bandits

COPY . .

RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.1/cmake-3.18.1.tar.gz \
    && tar -zxvf cmake-3.18.1.tar.gz \
    && cd cmake-3.18.1 \
    && ./bootstrap \
    && make \
    && make install \
    && cd .. \
    && rm -rf cmake-3.18.1 \
    && rm cmake-3.18.1.tar.gz

RUN wget http://sourceforge.net/projects/arma/files/armadillo-9.900.2.tar.xz \
    && tar -xvf armadillo-9.900.2.tar.xz \
    && cd armadillo-9.900.2 \
    && cmake . \
    && make \
    && make install \
    && cd .. \
    && rm -rf armadillo-9.900.2 \
    && rm armadillo-9.900.2.tar.xz

RUN mkdir build \
    && cd build \
    && cmake .. \
    && make
