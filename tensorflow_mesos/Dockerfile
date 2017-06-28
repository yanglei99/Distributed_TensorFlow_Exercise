
FROM       ubuntu:16.04

MAINTAINER Yang Lei <yanglei@us.ibm.com>

# The Tensorflow Image which can be used to submit job. T
#
# Folder structure:
#        Dockerfile
#        jobs
#            run.sh
#		 		the script that runs tensorflow job
#
# To build image once:
#
#        docker build -t tensorflow_job . 
#
# To run
#
#        docker run -d  tensorflow_job
#              run hello.py 
#
#        docker run -d -e TF_JOB_NAME=<your name>  -e TF_JOB_ARGUMENT=<your parameters> tensorflow_job
#			   run built-in python job 
#
#        docker run -d -v /shared_data:/train_dir -e TF_JOB_NAME=/train_dir/<location of a  python file>  -e TF_JOB_ARGUMENT=<your parameters> tensorflow_job
#			   run a python job shared through shared file system on the host /shared_data 
#

# reference https://www.tensorflow.org/install/

RUN apt-get update && apt-get install -y python-pip python-dev 
RUN pip install --upgrade pip
RUN pip install tensorflow

WORKDIR /jobs

ADD ./jobs/* /jobs/
RUN chmod +x *.sh

VOLUME /train_dir

ENV TF_CPP_MIN_LOG_LEVEL 2
ENV TF_JOB_NAME hello.py
#ENV TF_JOB_ARGUMENTS

CMD ["/jobs/run.sh"]

