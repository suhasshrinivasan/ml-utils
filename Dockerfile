### base image
FROM ghcr.io/walkerlab/docker-pytorch-cuda:cuda-11.8.0-pytorch-1.13.0-torchvision-0.14.0-torchaudio-0.13.0-ubuntu-20.04

RUN apt-get update
RUN pip3 install --upgrade pip

### build project as package
COPY . /src/project
RUN pip3 install --user -e /src/project