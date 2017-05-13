# reference https://www.tensorflow.org/deploy/distributed
# Usage:
# python dist_test.py --ps_hosts=127.0.0.1:2222 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=ps --task_index=0
# python dist_test.py --ps_hosts=127.0.0.1:2222 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=worker --task_index=0
# python dist_test.py --ps_hosts=127.0.0.1:2222 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=worker --task_index=1

import argparse
import sys
import numpy as np

import tensorflow as tf

FLAGS = None
        
def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):
		# Build model...
        x_data = np.random.rand(100).astype(np.float32)
        y_data = x_data * 0.1 + 0.3
        W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
        b = tf.Variable(tf.zeros([1]))
        y = W * x_data + b
        loss = tf.reduce_mean(tf.square(y - y_data))
        
        global_step = tf.contrib.framework.get_or_create_global_step()
        train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=50001)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir= FLAGS.log_dir,
                                           hooks=hooks) as mon_sess:
      import time
      ts = time.time()
      import datetime
      print('Start Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
      local_step = 0
      
      while not mon_sess.should_stop():
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.
        if local_step % 100 == 99:
          _, last_loss, last_global_step = mon_sess.run([train_op, loss, global_step ])
          print(local_step+1, last_global_step+1, last_loss)
        else:
          _, last_loss, last_global_step = mon_sess.run([train_op, loss, global_step])
        local_step += 1
      print('End Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
      print(local_step, last_global_step+1, time.time()-ts, last_loss)
 
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  parser.add_argument(
      "--log_dir", 
      type=str, 
      default='/tmp/train_logs/',
      help='Summaries log directory'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)