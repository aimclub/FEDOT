# This dockerfile is for running FEDOT on GPU
# Run following command from the project root directory to buld an image:
# user@projectpath: docker build -t mynewimage -f gpu/Dockerfile

FROM nvcr.io/nvidia/rapidsai/rapidsai:21.06-cuda11.2-base-ubuntu18.04

WORKDIR /home/FEDOT
COPY . /home/FEDOT

RUN apt-get update
RUN apt-get install -y wget

ENV PYTHONPATH /home/FEDOT

RUN pip install -r requirements.txt
