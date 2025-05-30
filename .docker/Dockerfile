FROM python:3.11-slim as base

# base
RUN apt-get update && apt-get install -y apt-utils default-jdk curl wget gcc python3-dev gnupg \
    software-properties-common && apt-get clean && rm -rf /var/lib/apt/lists/*

# spark
RUN wget https://archive.apache.org/dist/spark/spark-3.5.4/spark-3.5.4-bin-hadoop3.tgz && \
    tar xf spark-* && \
    mv spark-3.5.4-bin-hadoop3 /opt/spark && \
    rm spark-*.tgz

ENV SPARK_HOME "/opt/spark"
ENV HADOOP_HOME "${SPARK_HOME}"
ENV PATH "${PATH}:${SPARK_HOME}/bin:${SPARK_HOME}/sbin"
ENV PYSPARK_PYTHON "/usr/local/bin/python3"
ENV PYSPARK_DRIVER_PYTHON "${PYSPARK_PYTHON}"
ENV PYTHONPATH "${PYTHONPATH}:/local:/:"

COPY /sparkleframe /sparkleframe
ENV PYTHONPATH "${PYTHONPATH}:/local:/sparkleframe:/:"

# requirements
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt