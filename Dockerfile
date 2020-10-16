FROM quay.io/pypa/manylinux1_x86_64

RUN yum install -y openblas-devel \
    lapack-devel \
    arpack-devel \
    superlu-devel

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

RUN git clone https://gitlab.com/conradsnicta/armadillo-code.git \
    && cd armadillo-code \
    && cmake . \
    && make \
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

RUN mkdir /home/bandits \
    && cd /home/bandits

COPY . /home/bandits/
