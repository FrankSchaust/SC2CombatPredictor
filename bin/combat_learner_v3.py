#!/usr/bin/env python3
# Copyright 2017 Frank Schaust. All Rights Reserved.
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

import os
import sys

import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, Dropout, MaxPooling2D, Conv3D, MaxPooling3D
from absl import app

from bin.util import *
from lib.config import REPLAYS_PARSED_DIR
from data import simulation_pb2
from bin.load_batch import load_batch



tf.logging.set_verbosity(tf.logging.INFO)






def cnn_model_fn(features, labels, mode):
    depth = 20
    input_layer = tf.reshape(features["x"], [-1, depth, 84, 84, 1])
    
    conv1 = tf.layers.conv3d(
      inputs=input_layer,
      filters=32,
      kernel_size=[depth, 5, 5],
      activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[1, 4, 4], strides=(1,4,4))

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 20, 20, 32]
  # Output Tensor Shape: [batch_size, 16, 16, 64]
    conv2 = tf.layers.conv3d(
      inputs=pool1,
      filters=64,
      kernel_size=[1, 5, 5],
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 16, 16, 64]
  # Output Tensor Shape: [batch_size, 4, 4, 64]
    pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[1, 4, 4], strides=(1,4,4))

    pool2_flat = tf.reshape(pool2, [-1, 4 * 4 * 64])


    
    dense = tf.layers.dense(inputs=pool2_flat, units=126, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
      # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 3]
    logits = tf.layers.dense(inputs=dropout, units=3)
     
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(logits, 1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, label_smoothing=0.00000001)

  # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    
    
    
		

def main(unused_argv):
    # Set numpy print options so that numpy arrays containing feature layers
    # are printed completely. Useful for debugging.
    np.set_printoptions(threshold=(84 * 84), linewidth=(84 * 2 + 10))
    depth = 20
	# Loading example files
    replay_parsed_files = []
    replay_parsed_files = build_file_array(version='')
	
    # basic configurations for training:
    learning_rate = 0.0001
    epochs = 10
    batch_size = 10
    capped_batch = 1000
	
    #train_cnn(replay_parsed_files, learning_rate, epochs, batch_size, capped_batch, depth)
    
    # Load training and eval data
    x_train, x_test, y_train, y_test = load_batch(replay_parsed_files, 100, 0, True)  
    assert not np.any(np.isnan(x_train))
    assert not np.any(np.isnan(x_test))
    sess = tf.Session()
    print(sess.run(tf.one_hot(indices=tf.cast(y_test, tf.int32), depth=3)))
    # Create the Estimator
    classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    #print(x_train.shape, y_train.shape)
    #print(x_train.shape, y_train.shape)
  # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": x_train},
      y=y_train,
      batch_size=10,
      num_epochs=None,
      shuffle=True)
    classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": x_test},
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(y_test, eval_input_fn)
    print(eval_results)
    
    
    
	

     

def entry_point():
    app.run(main)


if __name__ == '__main__':
    app.run(main)
	 