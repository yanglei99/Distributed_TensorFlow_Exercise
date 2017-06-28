#!/bin/bash

env | grep TF

python $TF_JOB_NAME $TF_JOB_ARGUMENTS

echo "end of training" > end.log

tail -F end.log