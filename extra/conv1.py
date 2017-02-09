# Adapted from the TensorFlow Mnist tutorial (see license).

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

# Input images
x = tf.placeholder(tf.float32, shape=[None, 28*28])
# Output classes
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W,
                      strides=[1, 1, 1, 1],
                      padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='VALID')

x_image = tf.reshape(x, [-1, 28, 28, 1])

LEARNING_RATE = 0.03
BATCH_SIZE = 10
CONV1_CHANS = 2
FC_SIZE = 100

# First conv layer
W_conv1 = weight_variable([5, 5, 1, CONV1_CHANS])
b_conv1 = bias_variable([CONV1_CHANS])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# FC layer
W_fc1 = weight_variable([4 * 4 * CONV2_CHANS, FC_SIZE])
b_fc1 = bias_variable([FC_SIZE])

h_pool1_flat = tf.reshape(h_pool1, [-1, 4 * 4 * CONV2_CHANS])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

# Readout
W_fc2 = weight_variable([FC_SIZE, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

# Training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(BATCH_SIZE)
  if i % 100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:mnist.test.images, y_: mnist.test.labels})
    print("step %d, training accuracy %f"%(i*BATCH_SIZE, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels}))
