# This the base image for running FEDOT in container
FROM python:3.8

RUN mkdir -p /home/FEDOT
COPY . /home/FEDOT

WORKDIR /home/FEDOT

RUN apt-get update
RUN apt-get install -y wget
RUN /usr/local/bin/python -m pip install --upgrade pip

RUN pip3 install -r requirements.txt
