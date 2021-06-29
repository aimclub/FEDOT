FROM nvcr.io/nvidia/rapidsai/rapidsai:21.06-cuda11.2-base-ubuntu18.04
RUN mkdir -p /home/FEDOT
COPY ./FEDOT /home/FEDOT
