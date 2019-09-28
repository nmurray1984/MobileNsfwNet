FROM ubuntu:18.04
LABEL maintainer caffe-maint@googlegroups.com

RUN apt-get update

RUN export DEBIAN_FRONTEND=noninteractive && \
    ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get install -y python3 python3-pip wget

RUN apt-get install -y caffe-cpu

RUN pip3 install nsfw

RUN pip3 install pillow

RUN mkdir /datadrive

WORKDIR /workspace
