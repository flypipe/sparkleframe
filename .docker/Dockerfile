FROM apache/spark:3.5.4-java17-python3@sha256:3138f9251da619a20e702e7068c0a7ebf8f71c10d2d0395c4975c43c81367b2a AS base

USER root
# base
RUN apt-get update && apt-get install -y apt-utils curl wget graphviz gcc gnupg \
    software-properties-common && apt-get clean && rm -rf /var/lib/apt/lists/*

# use python3 as python
RUN ln -s $(which python3) /usr/local/bin/python

COPY /sparkleframe /sparkleframe
ENV PYTHONPATH "${PYTHONPATH}:/local:/sparkleframe:/:"

# requirements
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

WORKDIR /