FROM cppdebug:0.1
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
RUN git clone -j8 https://github.com/RUrlus/carma.git
RUN cd carma && git checkout origin/unstable && git submodule update --init
COPY build_run.sh /carma/build_run.sh
