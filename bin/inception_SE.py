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

import numpy as np
import tensorflow as tf
from absl import app

from bin.util import print_layer_details
from bin.modules import stem, inception_a, inception_b, inception_c, reduction_a, reduction_b, se_layer


def inception_v4_se(x_):
    #reduction ratio for se_layers
    rr = 4
    with tf.name_scope("Stem"):
        x_ = tf.layers.max_pooling3d(x_, pool_size=[1,2,2], strides=(1,2,2))
        x_ = stem(x_, "Stem")
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
    with tf.name_scope("Inception_A"):
        for i in range(1):
            x_ = inception_a(x_, "Inception_A")
            channel = int(np.shape(x_)[-1])
            x_ = se_layer(x_, out_dim=channel, ratio=rr, scope="SE_A"+str(i))
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
    with tf.name_scope("Reduction_A"):
        x_ = reduction_a(x_, "Reduction_A")
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
    with tf.name_scope("Inception_B"):
        for i in range(1):
            x_ = inception_b(x_, "Inception_B") 
            channel = int(np.shape(x_)[-1])
            x_ = se_layer(x_, out_dim=channel, ratio=rr, scope="SE_B"+str(i))
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
    with tf.name_scope("Reduction_B"):
        x_ = reduction_b(x_, "Reduction_B")
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
    with tf.name_scope("Inception_C"):
        for i in range(1):
            x_ = inception_c(x_, "Inception_C") 
            channel = int(np.shape(x_)[-1])
            x_ = se_layer(x_, out_dim=channel, ratio=rr, scope="SE_C"+str(i))
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
    with tf.name_scope("Final_Layer"):
        width = int(int(x_.get_shape()[2])/2)
        height = int(int(x_.get_shape()[3])/2)
        x_avg = tf.layers.average_pooling3d(x_, pool_size=[1, width, height], strides=(1,width, height))
        x_flat = tf.layers.flatten(inputs=x_avg)
        x_dense = tf.layers.dense(inputs=x_flat, units=64, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        print_layer_details(tf.contrib.framework.get_name_scope(), x_dense.get_shape())
        y_ = tf.layers.dense(inputs=x_dense, units=3, kernel_regularizer=tf.keras.regularizers.l2(1e-4))

    return y_