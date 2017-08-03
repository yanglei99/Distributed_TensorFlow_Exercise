# Distributed TensorFlow

This repository contains the exercise of Distributed TensorFlow run on Softlayer. 

## Distributed TensorFlow on Mesos/Marathon

The exercise is revised from [TensorFlow Ecosystem - Marathon](https://github.com/tensorflow/ecosystem/tree/master/marathon). It uses [Softlayer File Storage](https://knowledgelayer.softlayer.com/topic/file-storage) as shared file system.  

Reference [setting up Mesosphere DC/OS on Softlayer using Terraform, including GPU on BareMetal](https://github.com/yanglei99/terraform_softlayer/tree/master/dcos). 

Make sure `enable_file_storage=true` 

Here is a sample [runtime topology](images/tensorflow.jpg)

### Verified

* Mesosphere DC/OS: v1.9.2 (Docker v1.13.1)
* Tensorflow: v1.2.0


### Image

Follow instructions in [Dockerfile](Dockerfile) to build Docker Image. Here is the [pre-built image](https://hub.docker.com/r/yanglei99/tensorflow_job/)


### Run Tensorflow Job

#### Prepare the data 

To run mnist related training, follow [instruction](https://github.com/tensorflow/ecosystem/tree/master/docker) to convert data to records with[convert_to_records.py](https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/examples/how_tos/reading_data/convert_to_records.py). Then upload files to shared file system through one of the DC/OS agent (public or private) IP. The following assumes you are using the default value of `nfs_dir=/shared_data` to provision the File Storage for DC/OS

    # make sure you create the directory needed on the shared file system
    
	scp -i do-key /tmp/data/*.tfrecords root@$agentIp:/shared_data/dist_mnist/data/


#### Render the Marathon JSON definition

Get [render_template.py](https://github.com/tensorflow/ecosystem/blob/master/render_template.py).

##### For built-in python job
	
	python render_template.py template/dist_test.json.jinja > dist_test.json
	python render_template.py template/dist_test_summary.json.jinja > dist_test_summary.json
	python render_template.py template/dist_mnist.json.jinja > dist_mnist.json

##### For other python job 

Anther [MNIST Training](https://github.com/tensorflow/ecosystem/blob/master/docker/mnist.py)

Download the python file onto the shared file system through one of the agent node
    
    wget https://raw.githubusercontent.com/tensorflow/ecosystem/master/docker/mnist.py
    
    scp -i do-key mnist.py root@$agentIp:/shared_data/dist_mnist
 
Render the job python

    python render_template.py template/mnist.json.jinja > mnist.json
    

### Execute Distributed TensorFlow from Marathon JSON 

Make sure you have enough Agent to run jobs

	curl -XPUT -H 'Content-Type: application/json' -d @dist_test.json http://$marathonIp:8080/v2/groups
	curl -XPUT -H 'Content-Type: application/json' -d @dist_test_summary.json http://$marathonIp:8080/v2/groups
	curl -XPUT -H 'Content-Type: application/json' -d @dist_mnist.json http://$marathonIp:8080/v2/groups

	curl -XPUT -H 'Content-Type: application/json' -d @mnist.json http://$marathonIp:8080/v2/groups

	# to remove 
	
	curl -XDELETE http://$marathonIp:8080/v2/groups/dist.test
	curl -XDELETE http://$marathonIp:8080/v2/groups/dist.test.summary
	curl -XDELETE http://$marathonIp:8080/v2/groups/dist.mnist

	curl -XDELETE http://$marathonIp:8080/v2/groups/mnist


## GPU

Make sure your DCOS cluster is provisioned with GPU BareMetal Agent.

### Execute GPU TensorFlow Notebook from Marathon JSON 

	curl -XPUT -H 'Content-Type: application/json' -d @gpu/docker-gpu-tf.json http://$marathonIp:8080/v2/groups

You can use SSH tunnel to access the Notebook Web UI

	ssh -i do-key -L 9000:YOUR_BAREMETAL_IP:TF_NOTEBOOK_PORT root@YOUR_BAREMETAL_IP
	http://localhost:9000/?token=TF_NOTEBOOK_TOKEN
	
#### TensorBoard

Available at: http://tensorboard.marathon.mesos:6006

### Known Issues

