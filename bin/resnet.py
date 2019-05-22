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

import six
import keras
import numpy as np
import tensorflow as tf
from absl import app

from bin.util import *
from lib.config import REPLAYS_PARSED_DIR, REPO_DIR, STANDARD_VERSION
from data import simulation_pb2
from bin.modules import *

def resnet(x_):
    a = 5
    b = 3
    num_classes = 3
    repetitions = [a,b]
    reg_factor = 1e-4
    block_fn = basic_block
    
    block_fn = get_block(block_fn)
    with tf.name_scope("First_Layer"):
        conv1 = conv_bn_relu(x_, filters=64, kernel_size=[1, 7, 7], strides=(1,2,2), kernel_regularizer=tf.keras.regularizers.l2(reg_factor), padding='VALID')
        print_layer_details(tf.contrib.framework.get_name_scope(), conv1.get_shape())
        pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[1,3,3], strides=(1,2,2), padding='SAME')
    
    block = pool1
    #print(block.get_shape())
    filters = 64
    for i, r in enumerate(repetitions):
        with tf.name_scope("Residual_Block_"+str(i+1)):
            block = residual_block_3d(block, block_fn, filters=filters, repetitions=r, kernel_regularizer=tf.keras.regularizers.l2(reg_factor), is_first_layer=(i == 0), scope="Residual_Block_"+str(i+1))
            filters *= 2
        
    block_output = batch_norm(block)
    width = int(int(block.get_shape()[2])/2)
    height = int(int(block.get_shape()[3])/2)
    with tf.name_scope("Avg_Pooling"):
        pool2 = tf.layers.average_pooling3d(inputs=block_output, 
                                           pool_size=[1,
                                                      width,
                                                      height],
                                           strides=(1,width,height))
        print_layer_details(tf.contrib.framework.get_name_scope(), pool2.get_shape())            
    flatten1 = tf.layers.flatten(pool2)
    x_ = flatten1
    with tf.name_scope("Dense_Layer"):
        x_ = tf.layers.dense(inputs=x_, units=64, kernel_regularizer=tf.keras.regularizers.l2(reg_factor), activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
        
    y_ = tf.layers.dense(inputs=x_, units=num_classes)

    return y_

