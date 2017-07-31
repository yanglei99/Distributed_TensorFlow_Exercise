# reference https://github.com/hustcat/tensorflow_examples/blob/master/mnist_distributed/dist_fifo.py
# Usage:
# python dist_mnist.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225,127.0.0.1:2226 --job_name=ps --task_index=0
# python dist_mnist.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225,127.0.0.1:2226 --job_name=ps --task_index=1
# python dist_mnist.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225,127.0.0.1:2226 --job_name=worker --task_index=0
# python dist_mnist.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225,127.0.0.1:2226 --job_name=worker --task_index=1
# python dist_mnist.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225,127.0.0.1:2226 --job_name=worker --task_index=2


import math
import sys
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("hidden_units", 100,
                            "Number of units in the hidden layer of the NN")
tf.app.flags.DEFINE_string("data_dir", "/tmp/tensorflow/mnist/input_data/",
                           "Directory for storing mnist data")
tf.app.flags.DEFINE_string("log_dir", "/tmp/tensorflow/mnist/logs/",
                           "Directory for storing mnist train logs")
tf.app.flags.DEFINE_integer("batch_size", 100, "Training batch size")
tf.app.flags.DEFINE_integer("workers", 2, "Number of workers")
tf.app.flags.DEFINE_integer("ps", 1, "Number of ps")
tf.app.flags.DEFINE_integer("max_step", 2000, "Number of max steps")

FLAGS = tf.app.flags.FLAGS

IMAGE_PIXELS = 28

def create_done_queue(i):
  """Queue used to signal death for i'th ps shard. Intended to have 
  all workers enqueue an item onto it to signal doneness."""
  
  with tf.device("/job:ps/task:%d" % (i)):
    return tf.FIFOQueue(FLAGS.workers, tf.int32, shared_name="done_queue"+
                        str(i))
  
def create_done_queues():
  return [create_done_queue(i) for i in range(FLAGS.ps)]

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
    sess = tf.Session(server.target)
    queue = create_done_queue(FLAGS.task_index)
  
    # wait until all workers are done
    for i in range(FLAGS.workers):
      sess.run(queue.dequeue())
      print("ps %d received done %d" % (FLAGS.task_index, i))
     
    print("ps %d: quitting"%(FLAGS.task_index))
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Variables of the hidden layer
      hid_w = tf.Variable(
          tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                              stddev=1.0 / IMAGE_PIXELS), name="hid_w")
      hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

      # Variables of the softmax layer
      sm_w = tf.Variable(
          tf.truncated_normal([FLAGS.hidden_units, 10],
                              stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
          name="sm_w")
      sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

      x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
      y_ = tf.placeholder(tf.float32, [None, 10])

      hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
      hid = tf.nn.relu(hid_lin)

      y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
      loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

      global_step = tf.Variable(0)

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)
       
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      saver = tf.train.Saver()
      summary_op = tf.summary.merge_all()
      init_op = tf.global_variables_initializer()
     

      enq_ops = []
      for q in create_done_queues():
        qop = q.enqueue(1)
        enq_ops.append(qop)

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir= FLAGS.log_dir,
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=60)

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    begin_time = time.time()
    frequency = 100
    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target) as sess:
      # Loop until the supervisor shuts down or 100000 steps have completed.
      step = 0
      while not sv.should_stop() and step < FLAGS.max_step:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        start_time = time.time()

        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        train_feed = {x: batch_xs, y_: batch_ys}

        _, step = sess.run([train_op, global_step], feed_dict=train_feed)
        elapsed_time = time.time() - start_time
        if step % frequency == 0: 
            print ("Done step %d" % step, " AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))

      # Test trained model
      print("Test-Accuracy: %2.4f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

      # signal to ps shards that we are done
      #for q in create_done_queues():
      #sess.run(q.enqueue(1))
      for op in enq_ops:
        sess.run(op)
    print("Total Time: %3.2fs" % float(time.time() - begin_time))

    # Ask for all the services to stop.
    sv.stop()

if __name__ == "__main__":
  tf.app.run()