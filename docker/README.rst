Running FEDOT in Docker
============

Here are some dockerfiles to run FEDOT


Versions
=========

- **Dockerfile** Full version of FEDOT (fedot + fedot[extra]) for python 3.8
- **Dockerfile_light** Light version of FEDOT (fedot only) for python 3.8
- **GPU** A GPU version of FEDOT for python 3.8
- **Dockerfile_Jupiter** A Jupiter notebook version for python 3.10. Below you can find an instruction on how to run it under Linux.


Jupiter
=========
- **git clone** clone this repo
- **cd FEDOT** navigate to the root folder
- **cd docker** navigat to the docker folder
- **docker build -t jupyter-fedot-tst -f Dockerfile_Jupiter .** buid an image and give it a tag "jupyter-fedot"
- **docker run -p 8888:8888 -v $(pwd):/home/jovyan/work jupyter-with-fedot** run a container, it will be available under http://[YOUR_IP]:8888, here we use current dirrectory to store all files
- **copy the URL with a token and open in a browser** - if everything runs nornally you will see a link like ..  http://127.0.0.1:8888/lab?token=db8ce02fbed23c3ecd896408a494de176a70d73cf51e203f