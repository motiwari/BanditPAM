FROM quay.io/pypa/manylinux1_x86_64

RUN yum install -y openblas-devel \
    lapack-devel \
    arpack-devel \
    superlu-devel \
    git \
    vim-enhanced

RUN curl -LO https://github.com/squeaky-pl/centos-devtools/releases/download/6.2/gcc-6.2.0-binutils-2.27-x86_64.tar.bz2 \
    && tar xf gcc-6.2.0-binutils-2.27-x86_64.tar.bz2 \
    && rm -f gcc-6.2.0-binutils-2.27-x86_64.tar.bz2

RUN cd /home \
    && curl -LO https://cmake.org/files/v3.12/cmake-3.12.3.tar.gz \
    && tar zxf cmake-3.12.3.tar.gz \
    && cd cmake-3.12.3 \
    && ./bootstrap --prefix=/usr/local \
    && make \
    && make install \
    && cd .. \
    && rm -rf cmake-3.12.3 \
    && rm -f cmake-3.12.3.tar.gz

RUN ln -s /opt/devtools-6.2/bin/gcc /usr/bin/gcc \
    && ln -s /opt/devtools-6.2/bin/g++ /usr/bin/g++

RUN cd /home \
    && git clone https://gitlab.com/conradsnicta/armadillo-code.git \
    && git clone https://github.com/RUrlus/carma.git --recursive

ENV PATH=/usr/bin:/usr/local/bin:/opt/rh/devtoolset-2/root/usr/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin

ENV CC=/usr/bin/gcc

ENV CXX=/usr/bin/g++

ENV LD_LIBRARY_PATH=/opt/devtools-6.2/lib64:/opt/rh/devtoolset-2/root/usr/lib64:/opt/rh/devtoolset-2/root/usr/lib:/usr/local/lib64:/usr/local/lib

RUN cd /home/armadillo-code \
    && cmake . \
    && make install \
    && cd .. \
    && rm -rf armadillo-code

RUN yum -y install git-all \
    && export PATH=/usr/local/libexec/git-core:$PATH \
    &&  cd /home/carma \
    && git checkout 502ab50 \
    && git submodule update --init \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install \
    && cd ../.. \
    && rm -rf carma

RUN mkdir /home/bandits \
    && cd /home/bandits

COPY . /home/bandits/

WORKDIR /home/bandits
