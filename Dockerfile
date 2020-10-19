FROM quay.io/pypa/manylinux1_x86_64

RUN yum install -y openblas-devel \
    lapack-devel \
    arpack-devel \
    superlu-devel

RUN curl -LO https://github.com/squeaky-pl/centos-devtools/releases/download/6.2/gcc-6.2.0-binutils-2.27-x86_64.tar.bz2 \
    && tar xvf gcc-6.2.0-binutils-2.27-x86_64.tar.bz2 \
    && rm gcc-6.2.0-binutils-2.27-x86_64.tar.bz2

ENV PATH=/opt/devtools-6.2/bin/:/opt/rh/devtoolset-2/root/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN cd home \
    && curl -LO https://cmake.org/files/v3.12/cmake-3.12.3.tar.gz \
    && tar zxvf cmake-3.12.3.tar.gz \
    && cd cmake-3.12.3 \
    && ./bootstrap --prefix=/usr/local \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf cmake-3.12.3 \
    && rm cmake-3.12.3.tar.gz

ENV CC=/opt/devtools-6.2/bin/gcc

ENV CXX=/opt/devtools-6.2/bin/g++

RUN cd home \
    && git clone https://gitlab.com/conradsnicta/armadillo-code.git \
    && cd armadillo-code \
    && cmake . \
    && make install \
    && cd .. \
    && rm -rf armadillo-code

RUN git clone https://github.com/RUrlus/carma.git --recursive \
    && cd carma \
    && git submodule update --init \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install \
    && cd ../.. \
    && rm -rf carma

RUN /opt/python/cp35-cp35m/bin/pip install pybind11==2.5.0

RUN mkdir /home/bandits \
    && cd /home/bandits

COPY . /home/bandits/
