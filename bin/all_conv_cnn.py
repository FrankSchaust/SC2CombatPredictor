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
from absl import app

from bin.util import print_layer_details


def all_conv(x_):
    CONV = 6
    num_classes = 3
    # # # Summary
    # Total Number of Vars = 1.370.817
    # # # Layer A 
    # input  8 x 84 x 84 x 1
    # output 8 x 84 x 84 x 96
    # number of variables = 84192
    ### Conv1 = 1 x 3 x 3 x 96 + 96 = 960
    ### Conv2 = 96 x 3 x 3 x 96 + 96 = 83040
    ### Batchnorm = Filter * 2 = 192
    with tf.name_scope("Layer_A"):
        x_ = tf.layers.conv3d(inputs=x_, filters=96, kernel_size=[1, 3, 3], strides=(1,1,1), padding='same', activation=tf.nn.relu)
        x_ = tf.layers.conv3d(inputs=x_, filters=96, kernel_size=[1, 3, 3], strides=(1,1,1), padding='same', activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
        x_ = tf.layers.batch_normalization(inputs=x_, training=True)
    # # # Layer B 
    # input  8 x 84 x 84 x 96
    # output 8 x 41 x 41 x 96
    # number of variables = 83232
    ### Conv1 = 96 x 3 x 3 x 96 + 96 = 83040
    ### Batchnorm = Filter * 2 = 192
    with tf.name_scope("Layer_B"):
        x_ = tf.layers.conv3d(inputs=x_, filters=96, kernel_size=[1, 3, 3], strides=(1,2,2), activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
        x_ = tf.layers.batch_normalization(inputs=x_, training=True)
    # # # Layer C_1 & C_2
    # input  8 x 41 x 41 x 96
    # output 8 x 41 x 41 x 192
    # number of variables = 498.816
    ### Conv1 = 96 x 3 x 3 x 192 + 192 = 166080
    ### Conv2 = 192 x 3 x 3 x 192 + 192 = 331968
    ### Batchnorm = Filter * 4 = 768
    for n in range(2):
        with tf.name_scope("Layer_C_"+str(n)):
            x_ = tf.layers.conv3d(inputs=x_, filters=192, kernel_size=[1, 3, 3], strides=(1,1,1), padding='same', activation=tf.nn.relu)
            print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
            x_ = tf.layers.batch_normalization(inputs=x_, training=True)
    # # # Layer D 
    # input  8 x 41 x 41 x 192
    # output 8 x 20 x 20 x 192
    # number of variables = 332352
    ### Conv1 = 192 x 3 x 3 x 192 + 192 = 331968
    ### Batchnorm = Filter * 2 = 384
    with tf.name_scope("Layer_D"):
        x_ = tf.layers.conv3d(inputs=x_, filters=192, kernel_size=[1, 3, 3], strides=(1,2,2), activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
        x_ = tf.layers.batch_normalization(inputs=x_, training=True)
    # # # Layer E 
    # input  8 x 20 x 20 x 192
    # output 8 x 19 x 19 x 192
    # number of variables = 332352
    ### Conv1 = 192 x 3 x 3 x 192 + 192 = 331968
    ### Batchnorm = Filter * 2 = 384
    with tf.name_scope("Layer_E"):
        x_ = tf.layers.conv3d(inputs=x_, filters=192, kernel_size=[1, 3, 3], strides=(1,1,1), padding='same', activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
        x_ = tf.layers.batch_normalization(inputs=x_, training=True)
    # # # Layer F 
    # input  8 x 19 x 19 x 192
    # output 8 x 19 x 19 x 192
    # number of variables = 37440
    ### Conv1 = 192 x 1 x 1 x 192 + 192 = 37056
    ### Batchnorm = Filter * 2 = 384
    with tf.name_scope("Layer_F"):
        x_ = tf.layers.conv3d(inputs=x_, filters=192, kernel_size=[1, 1, 1], strides=(1,1,1), activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
        x_ = tf.layers.batch_normalization(inputs=x_, training=True)
    # # # Dim Red Layer
    # input  8 x 19 x 19 x 192
    # output 8 x 19 x 19 x 10
    # number of variables = 1950
    ### Conv1 = 192 x 1 x 1 x 10 + 10 = 1930
    ### Batchnorm = Filter * 2 = 20
    with tf.name_scope("Dimensional_Red_Layer"):
        x_ = tf.layers.conv3d(inputs=x_, filters=10, kernel_size=[1, 1, 1], strides=(1,1,1), padding='same', activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
        x_ = tf.layers.batch_normalization(inputs=x_, training=True)
    # # # Final Layer
    # output (?, 3)
    ### Dense Input = 8 * 2 * 1 * 10 = 160 Nodes
    ### FCL = 160 * 3 + 3 = 483
    with tf.name_scope("Final_Layer"):
        width = int(int(x_.get_shape()[2])/2)
        height = int(int(x_.get_shape()[3])/2)
        x_avg = tf.layers.average_pooling3d(x_, pool_size=[1, width, height], strides=(1,width, height))
        x_ = tf.layers.flatten(inputs=x_avg)
        y_ = tf.layers.dense(inputs=x_, units=num_classes)
        print_layer_details(tf.contrib.framework.get_name_scope(), y_.get_shape())
        return y_

