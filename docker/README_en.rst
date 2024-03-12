Running FEDOT in Docker
=======================

Here are some dockerfiles to run FEDOT


Versions
========

- **Dockerfile** Full version of FEDOT (fedot + fedot[extra]) for python 3.8
- **Dockerfile_light** Light version of FEDOT (fedot only) for python 3.8
- **GPU** A GPU version of FEDOT for python 3.8
- **Dockerfile_Jupiter** A Jupiter notebook version for python 3.10. Below you can find an instruction on how to run it under Linux.


Jupiter
=======

- **check docker (docker-compose)** docker (docker-compose) should be installed
- `git clone https://github.com/aimclub/FEDOT.git` clone this repo
- `cd FEDOT` navigate to the root folder
- `cd docker/jupiter` navigate to the docker folder with jupiter notebook files

1. Run using docker-compose

- `docker-compose up` or `docker compose up`
- **copy the URL with a token and open in a browser** - if everything runs nornally you will see a link like `http://127.0.0.1:8888/lab?token=db8ce02fbed23c3ecd896408a494de176a70d73cf51e203f`

2. Run using docker

- `docker build -t jupyter-fedot -f Dockerfile_Jupiter .` buid an image and give it a tag "jupyter-fedot"
- `docker run -p 8888:8888 -v $(pwd):/home/jovyan/work jupyter-fedot` run a container, it will be available under `http://[YOUR_IP]:8888`, here we use current dirrectory to store all files
- **copy the URL with a token and open in a browser** - if everything runs nornally you will see a link like `http://127.0.0.1:8888/lab?token=db8ce02fbed23c3ecd896408a494de176a70d73cf51e203f`