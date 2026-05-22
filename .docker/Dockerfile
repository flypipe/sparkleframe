FROM docker.io/apache/spark:4.1.2-scala2.13-java17-python3-ubuntu@sha256:7f44fcdd38baa7bb6fdf97f84bc12d282655d8258001a9f60287a70fb9e5033e

# Base image ships Python 3.10; install 3.11 for parity with local / CI toolchains.
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Pip for 3.11 (image pip targets the default 3.10 interpreter).
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Prefer 3.11 when tools invoke `python` / `python3` from PATH (e.g. /usr/local/bin first).
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

COPY /sparkleframe /sparkleframe
ENV PYTHONPATH "${PYTHONPATH}:/local:/sparkleframe:/:"

# requirements
COPY requirements-dev.txt .
RUN python3.11 -m pip install -r requirements-dev.txt

WORKDIR /