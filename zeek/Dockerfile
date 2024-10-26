# Use the official Zeek base image
FROM zeek/zeek:latest

# Install build dependencies
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git cmake make gcc g++ libssl-dev libsasl2-dev libz-dev \
    python3-dev python3-pip swig curl libpcap-dev

# Install librdkafka from source
RUN curl -L https://github.com/edenhill/librdkafka/archive/v1.4.4.tar.gz | tar xvz && \
    cd librdkafka-1.4.4/ && \
    ./configure --enable-sasl && \
    make && \
    make install && \
    ldconfig

# Set LD_LIBRARY_PATH and LIBRDKAFKA_ROOT
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
ENV LIBRDKAFKA_ROOT=/usr/local

# Verify Zeek installation before installing the plugin
RUN zeek -N

# Install Zeek-Kafka plugin compatible with Zeek 4.0.5
RUN git clone https://github.com/SeisoLLC/zeek-kafka.git /opt/zeek-kafka && \
    cd /opt/zeek-kafka && \
    git checkout v1.2.0 && \
    ./configure --with-librdkafka=$LIBRDKAFKA_ROOT && \
    make && \
    make install && \
    ldconfig

# Verify the plugin is installed
RUN zeek -N Seiso::Kafka

# Copy Zeek scripts
COPY zeek_scripts/ /usr/local/zeek/share/zeek/site/

# Expose Zeek log directory
VOLUME /usr/local/zeek/logs

# Set working directory
WORKDIR /usr/local/zeek

# Default command (can be overridden)
CMD ["/bin/bash"]
