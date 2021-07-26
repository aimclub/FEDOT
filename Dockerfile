# This dockerfile is for running FEDOT on GPU
FROM nvcr.io/nvidia/rapidsai/rapidsai:21.06-cuda11.2-base-ubuntu18.04

RUN mkdir -p /home/FEDOT
COPY . /home/FEDOT

RUN apt-get update
RUN apt-get install -y wget

RUN pip install -r requirements.txt
