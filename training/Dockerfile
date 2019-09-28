FROM ubuntu:18.04
LABEL maintainer nmurray1984

RUN apt-get update

RUN export DEBIAN_FRONTEND=noninteractive && \
    ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get install -y python3 python3-pip wget

RUN pip3 install tensorflow

RUN pip3 install pillow

RUN pip3 install scipy

RUN pip3 install pandas

RUN mkdir /datadrive

WORKDIR /workspace
