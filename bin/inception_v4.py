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
from bin.modules import stem, inception_a, inception_b, inception_c, reduction_a, reduction_b

# adaption of the inception v4 architecture from szegedy et al. 
# inception and reduction modules are defined in bin/modules 
    ### Modifications for this work ###
    # - filter counts are reduced by the factor 4 
    # - as a result the corresponding trainable variables are reduced from ~32 million to 1.3 million to achieve comparability across all architectures
    # - every module occurs only once within the architecture
    # - Identity shortcuts are defined to skip every module
    # - Reduction conv layer introduced before the input is flattened for dense computation, to reduce dimensionality and therefore the variables held by the dense layer
    # - instead of using only one dense layer with the number of classes as units we introduce an additional dense layer to achieve high level reasoning
    # - The average pooling layer does not average across the whole matrix, but instead averages across each armies zone to compute 2 comparable values of each feature matrix for the dense layer(shape 8,2,1)

def inception_v4(x_):
    with tf.name_scope("Stem"):
        x_max = tf.layers.max_pooling3d(x_, pool_size=[1,2,2], strides=(1,2,2))
        x_ = stem(x_max, "Stem")
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
    with tf.name_scope("Inception_A"):
        shortcut = x_
        for i in range(1):
            x_ = inception_a(x_, "Inception_A"+str(i))
            # map identity with [1,1,1] kernel to match dimensions
            shortcut = tf.layers.conv3d(shortcut, filters=x_.get_shape()[4], kernel_size=[1,1,1], strides=(1,1,1), padding='Valid', kernel_regularizer=tf.keras.regularizers.l2(1e-4), activation=tf.nn.relu)
            # [shortcut matrix, residual matrix]
            x_ = tf.keras.layers.add([shortcut, x_])
            x_ = tf.nn.relu(x_)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
    with tf.name_scope("Reduction_A"):
        x_ = reduction_a(x_, "Reduction_A")
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
    with tf.name_scope("Inception_B"):
        shortcut = x_
        for i in range(1):
            x_ = inception_b(x_, "Inception_B")
            shortcut = tf.layers.conv3d(shortcut, filters=x_.get_shape()[4], kernel_size=[1,1,1], strides=(1,1,1), padding='Valid', kernel_regularizer=tf.keras.regularizers.l2(1e-4), activation=tf.nn.relu)
            x_ = tf.keras.layers.add([shortcut, x_])
            x_ = tf.nn.relu(x_)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
    with tf.name_scope("Reduction_B"):
        x_ = reduction_b(x_, "Reduction_B")
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
    with tf.name_scope("Inception_C"):
        shortcut = x_
        for i in range(1):
            x_ = inception_c(x_, "Inception_C")
            shortcut = tf.layers.conv3d(shortcut, filters=x_.get_shape()[4], kernel_size=[1,1,1], strides=(1,1,1), padding='Valid', kernel_regularizer=tf.keras.regularizers.l2(1e-4), activation=tf.nn.relu)
            x_ = tf.keras.layers.add([shortcut, x_])
            x_ = tf.nn.relu(x_)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
    # with tf.name_scope("Reduction_Conv"):
    #     x_ = tf.layers.conv3d(x_, filters=32, kernel_size=[1,1,1], strides=(1,1,1), padding='Valid', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
    #     print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
    with tf.name_scope("Final_Layer"):
        width = int(int(x_.get_shape()[2])/2)
        height = int(int(x_.get_shape()[3])/2)
        x_avg = tf.layers.average_pooling3d(x_, pool_size=[1, width, height], strides=(1,width,height))
        x_flat = tf.layers.flatten(inputs=x_avg)
        x_dense = tf.layers.dense(inputs=x_flat, units=64, activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_dense.get_shape())
        y_ = tf.layers.dense(inputs=x_dense, units=3)
    return y_