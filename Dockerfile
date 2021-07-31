FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends tzdata
ENV TZ Asia/Tokyo

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    vim \
    graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


ENV HOME /src
WORKDIR /src
ENV SHELL /bin/bash


ADD requirements.txt /src
RUN pip install -r requirements.txt

EXPOSE 36000
EXPOSE 36001
EXPOSE 36002