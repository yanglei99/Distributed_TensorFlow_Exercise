# Distributed TensorFlow

This repository contains the exercise of Distributed TensorFlow run on Softlayer.

Reference [setting up Mesosphere DC/OS on Softlayer using Terraform](https://github.com/yanglei99/terraform_softlayer/tree/master/dcos)


## Distributed TensorFlow on Mesos/Marathon

Reference [TensorFlow Ecosystem - Marathon](https://github.com/tensorflow/ecosystem/tree/master/marathon)

### Prereq

#### Docker Image

Follow [instruction](https://github.com/tensorflow/ecosystem/tree/master/docker) to build Docker Image from [Dockerfile.hdfs](Dockerfile.hdfs)

	docker build -t tensorflow_test -f Dockerfile.hdfs .

#### HDFS

As HDFS is needed for sharded checkpoint and tensorboard, you can either use the DC/OS HDFS support or start HDFS using [docker](https://github.com/sequenceiq/hadoop-docker).

	curl -i -H 'Content-Type: application/json' -d@marathon-hdfs.json $marathonIp:8080/v2/apps
	
HDFS console at: http://$HDFS_HOST:50070/

To run nmist, follow [instruction](https://github.com/tensorflow/ecosystem/tree/master/docker) to convert data to record and upload to HDFS

	sudo hadoop fs -put -f /tmp/data/train.tfrecords hdfs://$HDFS_HOST:9000/train_dir/mnist_data/train.tfrecords
	sudo hadoop fs -put -f /tmp/data/test.tfrecords hdfs://$HDFS_HOST:9000/train_dir/mnist_data/test.tfrecords
	sudo hadoop fs -put -f /tmp/data/validation.tfrecords hdfs://$HDFS_HOST:9000/train_dir/mnist_data/validation.tfrecords
	

Docker Image is built
### Render the Marathon JSON definition

	python render_template.py dist.test.json.jinja > dist.test.json
	python render_template.py dist.test.summary.json.jinja > dist.test.summary.json
	python render_template.py mnist.json.jinja > mnist.json


### Execute Distributed TensorFlow from Marathon JSON 


	curl -XPUT -H 'Content-Type: application/json' -d @dist.test.json http://$marathonIp:8080/v2/groups
	curl -XPUT -H 'Content-Type: application/json' -d @dist.test.summary.json http://$marathonIp:8080/v2/groups
	curl -XPUT -H 'Content-Type: application/json' -d @mnist.json http://$marathonIp:8080/v2/groups


### TensorBoard

	http://tensorboard.marathon.mesos:6006
