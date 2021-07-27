# This the base image for running FEDOT in container
FROM ubuntu:20.04

WORKDIR /home/FEDOT
COPY . /home/FEDOT

RUN apt-get update
RUN apt-get install -y python3.8 python3-pip
RUN pip3 install pip==19.3.1

ENV PYTHONPATH /home/FEDOT

RUN pip install --trusted-host pypi.python.org -r requirements.txt
