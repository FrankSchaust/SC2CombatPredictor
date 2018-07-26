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

from lib.config import REPLAYS_PARSED_DIR
from data import simulation_pb2
from bin.load_batch import load_batch

def main():
    # Set numpy print options so that numpy arrays containing feature layers
    # are printed completely. Useful for debugging.
    np.set_printoptions(threshold=(84 * 84), linewidth=(84 * 2 + 10))
    depth = 20
	# Loading example files
    replay_parsed_files = []
    print("Creating list of used files")
    for root, dir, files in os.walk(REPLAYS_PARSED_DIR):
        for file in files:
            if file.endswith(".SC2Replay_parsed.gz"):
                replay_parsed_files.append(os.path.join(root, file))
    print("Available Files: ", len(replay_parsed_files))
	
    # basic configurations for training:
    learning_rate = 0.0001
    epochs = 10
    batch_size = 10
    capped_batch = 300
    num_classes = 2
    
    run_cnn(replay_parsed_files, learning_rate, epochs, batch_size, capped_batch, depth, num_classes)
    

def run_cnn(replay_parsed_files, learning_rate, epochs, batch_size, capped_batch, depth, num_classes):
    acc = 0
    t_acc = 0
    x = tf.placeholder(tf.float32, shape=[None, depth, 84, 84, 1])
    y = tf.placeholder(tf.float32, shape=[None, num_classes])
    
    #xs_train, xs_test, ys_train, ys_test = load_batch(replay_parsed_files, capped_batch, 0, True)
    
    # # Input 20, 84, 84 
    # # Output 20, 21, 21
    with tf.name_scope("layer_a"):
        conv1 = tf.layers.conv3d(inputs=x, filters=64, kernel_size=[1, 5, 5], name="conv1", padding='same', activation=tf.nn.relu)
        maxpool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[1, 4, 4], strides=(1,4,4))
        conv1_bn = tf.layers.batch_normalization(inputs=maxpool1, training=True)
    # # Input 20, 21, 21 
    # # Output 20, 11, 11
    with tf.name_scope("layer_b"):
        conv2 = tf.layers.conv3d(inputs=conv1_bn, filters=128, kernel_size=[1, 5, 5], name="conv2", padding='same', activation=tf.nn.relu)
        maxpool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[1, 4, 4], strides=(1,4,4))
        conv3_bn = tf.layers.batch_normalization(inputs=maxpool2, training=True)
    # # # Input 20, 11, 11
    # # # Output 20, 6, 6
    # with tf.name_scope("layer_c"):
        # conv3 = tf.layers.conv3d(inputs=conv3_bn, filters=256, kernel_size=[1, 5, 5], name="conv3", padding='same', activation=tf.nn.relu)
        # maxpool3 = tf.layers.max_pooling3d(inputs=conv3, pool_size=[1, 2, 2], strides=2)
    # with tf.name_scope("batch_norm"):
        # cnn3d_bn = tf.layers.batch_normalization(inputs=maxpool3, training=True)
    with tf.name_scope("layer_d"):
        dropout1 = tf.layers.dropout(inputs=conv3_bn, rate=0.75)
        flatten1 = tf.layers.flatten(inputs=dropout1)
        dense1 = tf.layers.dense(inputs=flatten1, units=256, name="dense1", activation=tf.nn.relu)
        dense1_bn = tf.layers.batch_normalization(inputs=dense1, training=True)
        dropout2 = tf.layers.dropout(inputs=dense1_bn, rate=0.5)
        y_conv = tf.layers.dense(inputs=dropout2, units=num_classes)
        
    y_ = tf.nn.softmax(y_conv)
       
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=y))

    # add an optimiser
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # setup the initialisation operator
    init_op = tf.global_variables_initializer()

    tf.summary.scalar('accuracy', acc)
    tf.summary.scalar('acc_test', t_acc)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('E:\\Users\\Frank\\Documents\\Documents\\Universitaet\\Bachelorarbeit')
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        # setup recording variables
        # add a summary to store the accuracy
        li = 0
        lis = 0
        total_batch = int(2000 / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            acc = 0
            t_acc = 0
            ys_test = []
            xs_test = []
            for i in range(total_batch):
                batch_x, _, batch_y, _, li = load_batch(replay_parsed_files, batch_size, i, li, True, True)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                train_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                sys.stdout.write("\r[%-20s] %.2f%% --- Batch %d from %d" % ('='*int(((i+1)/total_batch)*20), ((i+1)/total_batch)*100, i+1, total_batch))
                sys.stdout.flush()
                avg_cost += c / total_batch
                acc += train_acc / total_batch
            for i in range(total_batch): 
                _, batch_x, _, batch_y, lis = load_batch(replay_parsed_files, batch_size, i, lis, True, True)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                test_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                #sys.stdout.write("\r[%-20s] %.2f%% --- Batch %d from %d" % ('='*int(((i+1)/total_batch)*20), ((i+1)/total_batch)*100, i+1, total_batch))
                #sys.stdout.flush()
                #avg_cost += c / total_batch
                #xs_test.append(batch_x)
                #ys_test.append(batch_y)
                t_acc += test_acc / total_batch
            print(" --- Result of Epoch:", (epoch + 1), "Train accuracy: {:.3f}".format(acc), "cost: {:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(t_acc))
            merged = tf.summary.merge_all()
            summary = sess.run(merged)
            writer.add_summary(summary, epoch)

        print("\nTraining complete!")
        #writer.add_graph(sess.graph)
        #print(sess.run(accuracy, feed_dict={x: xs_test, y: ys_test}))

def getKernelAndBias(names):
    gr = tf.get_default_graph()
    kernels = []
    biases = []
    for n in names:
        kernel_val = gr.get_tensor_by_name(n + '/kernel:0').eval()
        bias_val = gr.get_tensor_by_name(n + '/bias:0').eval()
        kernels.append(kernel_val)
        biases.append(bias_val)
    return kernels, biases
if __name__ == "__main__":
    main()