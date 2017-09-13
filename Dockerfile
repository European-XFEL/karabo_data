# ubuntu:latest seems to be LTS, i.e. 16.04 at the moment
FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3 python3-pip python3-pytest
# use /io to mount host file system later
RUN mkdir /io
WORKDIR /io
