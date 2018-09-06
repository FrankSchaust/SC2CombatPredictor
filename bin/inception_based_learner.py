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

def main():
    
    depth = 13
	# Loading example files
    replay_parsed_files = []
    replay_parsed_files = build_file_array(version='1_3a')
	
    # basic configurations for training:
    learning_rate = 0.005
    epochs = 30
    batch_size = 10
    capped_batch = 300
    num_classes = 3
    batches = int(len(replay_parsed_files) / batch_size)
    run_inception(replay_parsed_files, learning_rate, epochs, batches, batch_size, capped_batch, depth, num_classes)

def inception(inputs, kernels=[1,1,1,1,1,1]):
    # far left side of inception module
    inception_1x1 = tf.layers.conv3d(inputs=inputs, filters=kernels[0], kernel_size=[1, 1, 1], strides=(1,1,1), padding='same', activation=tf.nn.relu)
    # near left side
    inception_red_3x3 = tf.layers.conv3d(inputs=inputs, filters=kernels[1], kernel_size=[1, 1, 1], strides=(1,1,1), padding='same', activation=tf.nn.relu)
    inception_3x3 = tf.layers.conv3d(inputs=inception_red_3x3, filters=kernels[2], kernel_size=[1, 3, 3], strides=(1,1,1), padding='same', activation=tf.nn.relu)
    # near right side
    inception_red_5x5 = tf.layers.conv3d(inputs=inputs, filters=kernels[3], kernel_size=[1, 1, 1], strides=(1,1,1), padding='same', activation=tf.nn.relu)
    inception_5x5 = tf.layers.conv3d(inputs=inception_red_5x5, filters=kernels[4], kernel_size=[1, 5, 5], strides=(1,1,1), padding='same', activation=tf.nn.relu)
    # far right side
    inception_max_pool = tf.layers.max_pooling3d(inputs=inputs, pool_size=[1, 2, 2], strides=(1,1,1))
    inception_max_pool_1x1 = tf.layers.conv3d(inputs=inputs, filters=kernels[5], kernel_size=[1, 1, 1], strides=(1,1,1), padding='same', activation=tf.nn.relu)
    inception = tf.concat([inception_1x1, inception_3x3, inception_5x5, inception_max_pool_1x1], 4)
    return inception
    
def run_inception(replay_parsed_files, learning_rate, epochs, batches, batch_size, capped_batch, depth, num_classes):
    acc = 0
    t_acc = 0
    x = tf.placeholder(tf.float32, shape=[None, depth, 84, 84, 1])
    y = tf.placeholder(tf.float32, shape=[None, num_classes])
    
    # # # Layer A 
    # convoluitional layer with 6x6 kernel, 2 stride and 64 kernels
    # max pool is 3x3 with 2 stride
    # input  13 x 84 x 84 x 1
    # output 13 x 21 x 21 x 64
    with tf.name_scope("Layer_A"):
        conv1 = tf.layers.conv3d(inputs=x, filters=64, kernel_size=[1, 6, 6], strides=(1,1,1), padding='same', activation=tf.nn.relu)
        #max_pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[1, 3, 3], strides=(1,2,2))
        bn1 = tf.layers.batch_normalization(inputs=conv1, training=True)
    # # # Layer B 
    # convolutional layer with 1x1 kernel used as reduction
    # convolutional layer with 3x3 kernel, 1 stride and 192 kernels 
    # max pool is 3x3 with 2 stride
    # input  13 x 21 x 21 x 64
    # output 13 x 10 x 10 x 192 
    with tf.name_scope("Layer_B"):
        conv_red_2 = tf.layers.conv3d(inputs=bn1, filters=64, kernel_size=[1, 1, 1], strides=(1,1,1), padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv3d(inputs=conv_red_2, filters=192, kernel_size=[1, 3, 3], strides=(1,1,1), padding='same', activation=tf.nn.relu)
        max_pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[1, 3, 3], strides=(1,2,2))
        bn2 = tf.layers.batch_normalization(inputs=max_pool2, training=True)
    # # # Inception A and B --- First inception layer
    # input  13 x 10 x 10 x 192
    # output 13 x 10 x 10 x 256
    with tf.name_scope("Inception_A_B"):
        inception_A = inception(bn2, [64, 96, 128, 16, 32, 32])  
        inception_B = inception(inception_A, [128, 128, 192, 32, 96, 64])
        bn3 = tf.layers.batch_normalization(inputs=inception_B, training=True)
    # # # Max Pool 3 x 3 Strides 2
    # input  13 x 10 x 10 x 480
    # output 13 x  5 x  5 x 480
    with tf.name_scope("Max_Pooling_Single_1"):
        max_pool3 = tf.layers.max_pooling3d(inputs=bn3, pool_size=[1, 2, 2], strides=(1,2,2))
    # # # Inception C to G second inception layer
    # input  13 x 5 x 5 x 480
    # output 13 x 5 x 5 x 832
    # with tf.name_scope("Inception_C_G"):
        # inception_C = inception(max_pool3, [192, 96, 208, 16, 48, 64])
        # inception_D = inception(inception_C, [160, 112, 224, 24, 64, 64])
        # inception_E = inception(inception_D, [128, 128, 256, 24, 64, 64])
        # inception_F = inception(inception_E, [112, 144, 288, 32, 64, 64])
        # inception_G = inception(inception_F, [256, 160, 320, 32, 128, 128])
    # # # Max Pool 3 x 3 Strides 2
    # input  13 x 5 x 5 x 832
    # output 13 x 2 x 2 x 832
    # with tf.name_scope("Max_Pooling_Single_2"):
        # max_pool4 = tf.layers.max_pooling3d(inputs=inception_G, pool_size=[1, 2, 2], strides=(1,2,2))
    # # # Inception H to I
    # input  13 x 2 x 2 x 832
    # output 13 x 2 x 2 x 1024
    # with tf.name_scope("Inception_H_I"):
        # inception_H = inception(max_pool3, [256, 160, 320, 32, 128, 128])
        # inception_I = inception(inception_H, [384, 192, 384, 48, 128, 128])
        # bn4 = tf.layers.batch_normalization(inputs=inception_I, training=True)
    # # # Last layer is and average pooling layer that reduces the dimensions to 1 x 1 x Y combined
    # with a fully connected layer  that concludes to the specified number of classes (3 in our case)
    # output 1 x 1 x 1 x 3
    with tf.name_scope("Last_Layer"):
        avg_pool = tf.layers.average_pooling3d(bn3, pool_size=[13, 1, 1], strides=(1,1,1))
        # print(avg_pool.get_shape())
        flatten = tf.layers.flatten(inputs=avg_pool)
        fully_connected_1000 = tf.layers.dense(inputs=flatten, units=256)
        fully_connected = tf.layers.dense(inputs=fully_connected_1000, units=num_classes)
    print(fully_connected.get_shape())
    
    softmax = tf.nn.softmax(fully_connected)
    
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fully_connected, labels=y))

    # add an optimiser
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(softmax, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # setup the initialisation operator
    init_op = tf.global_variables_initializer()

    writer = tf.summary.FileWriter('E:\\Users\\Frank\\Documents\\Documents\\Universitaet\\Bachelorarbeit')
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        # setup recording variables
        # add a summary to store the accuracy
        for epoch in range(epochs):
            avg_cost = 0
            acc = 0
            t_acc = 0
            ys_test = []
            xs_test = []
            li = 0
            lis = 0
            last_batch_acc = 0
            batches = 40
            for i in range(batches):
                batch_x, _, batch_y, _, li = load_batch(replay_parsed_files, batch_size, i, li, True)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                train_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                sys.stdout.write("\r[%-20s] %.2f%% --- Batch %d from %d --- Latest Acc: %.2f%%" % ('='*int(((i+1)/batches)*20), ((i+1)/batches)*100, i+1, batches, train_acc*100))
                sys.stdout.flush()
                avg_cost += c / batches
                acc += train_acc / batches
            for i in range(batches): 
                _, batch_x, _, batch_y, lis = load_batch(replay_parsed_files, batch_size, i, lis, True)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                test_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                #sys.stdout.write("\r[%-20s] %.2f%% --- Batch %d from %d" % ('='*int(((i+1)/total_batch)*20), ((i+1)/total_batch)*100, i+1, total_batch))
                #sys.stdout.flush()
                #avg_cost += c / total_batch
                #xs_test.append(batch_x)
                #ys_test.append(batch_y)
                t_acc += test_acc / batches
            print(" --- Result of Epoch:", (epoch + 1), "Train accuracy: {:.3f}".format(acc), "cost: {:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(t_acc))
            tf.summary.scalar('cost', avg_cost)
            tf.summary.scalar('Train Acc', acc)
            tf.summary.scalar('Test Acc', t_acc)
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
    