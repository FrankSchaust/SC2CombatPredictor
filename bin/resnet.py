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
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, Dropout, MaxPooling2D, Conv3D, MaxPooling3D
from absl import app

from bin.util import *
from lib.config import REPLAYS_PARSED_DIR, REPO_DIR, STANDARD_VERSION
from data import simulation_pb2
from bin.load_batch import load_batch
from bin.modules import *



def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    versions = ['1_3d_10sup']
    i = 0
    learning_rates = [0.001, 0.0005]
    ### constant declarations, the different architectures will be iterated by chosing different learning rates and ratios of convolutions to fully connected layers
    epochs = 50
    batch_size = 100
    capped_batch = 5000
    num_classes = 3
    depth = 13
    
	# Loading example files
    replay_parsed_files = []
    replay_parsed_files = build_file_array(version=versions)
    
    for lr in learning_rates:        
            # build the folder structure for tensorboard logs
            base_dir = os.path.join(REPO_DIR, 'tensorboard_logs', 'ResNet', 'AdamOpt', 'LearningRate_'+str(lr)+'_Repetitions_5_3_SampleSize_'+str(capped_batch))
            os.makedirs(base_dir, exist_ok=True)
            sub_dirs = get_immediate_subdirectories(base_dir)
            last_run_fin = get_number_of_last_run(base_dir, sub_dirs)
            # every structure will be trained 10 times
            for n in range(1):
                last_run_fin += 1
                tensorboard_dir = os.path.join(base_dir, 'Run '+str(last_run_fin))
                os.makedirs(tensorboard_dir, exist_ok=True)
                print('Model and Logs saved at %s' % (tensorboard_dir))
                run_cnns(replays=replay_parsed_files, lr=lr, epochs=epochs, batch_size=batch_size, capped_batch=capped_batch, tensorboard_dir=tensorboard_dir, versions=versions)
                
def run_cnns(replays=[], lr=0.5, epochs=15, batch_size=10, capped_batch=100, depth=13, num_classes=3, tensorboard_dir="", versions=STANDARD_VERSION):
    acc = 0
    t_acc = 0
    x = tf.placeholder(tf.float32, shape=[None, depth, 84, 84, 1])
    y = tf.placeholder(tf.float32, shape=[None, num_classes])
    reg_factor = 1e-4
    repetitions = [5,3]
    block_fn = basic_block
    
    block_fn = get_block(block_fn)
    x_ = remove_zero_layers(x)
    with tf.name_scope("First_Layer"):
        conv1 = conv_bn_relu(x_, filters=16, kernel_size=[1, 7, 7], strides=(1,2,2), kernel_regularizer=tf.keras.regularizers.l2(reg_factor), padding='VALID')
        param = print_layer_details(tf.contrib.framework.get_name_scope(), conv1.get_shape())
    # pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[3,3,3], strides=(2,2,2), padding='SAME')
    
    block = conv1
    #print(block.get_shape())
    filters = 16
    for i, r in enumerate(repetitions):
        with tf.name_scope("Residual_Block_"+str(i+1)):
            block = residual_block_3d(block, block_fn, filters=filters, repetitions=r, kernel_regularizer=tf.keras.regularizers.l2(reg_factor), is_first_layer=(i == 0), scope="Residual_Block_"+str(i+1))
            filters *= 2
        
    block_output = batch_norm(block)
    width = int(int(block.get_shape()[2])/2)
    height = int(int(block.get_shape()[3])/2)
    pool2 = tf.layers.average_pooling3d(inputs=block_output, 
                                        pool_size=[1,
                                                   width,
                                                   height],
                                        strides=(1,width,height))
    flatten1 = tf.layers.flatten(pool2)
    x_ = flatten1
    for i in range(4):
        x_ = tf.layers.dense(inputs=x_, units=(i+1)*5, kernel_regularizer=tf.keras.regularizers.l2(reg_factor))
        
    y_ = tf.layers.dense(inputs=x_, units=num_classes, kernel_regularizer=tf.keras.regularizers.l2(reg_factor))
    
    softmax = tf.nn.softmax(y_)

    #get trainable params
    para = get_params(tf.trainable_variables())
    print(para)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y))

    # add an optimiser
    optimiser = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(softmax, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # setup the initialisation operator
    init_op = tf.global_variables_initializer()
    # setup the save and restore functionality for variables 
    saver = tf.train.Saver()
    run_cnn(replays, lr, epochs, batch_size, capped_batch, depth, num_classes, optimiser, cross_entropy, accuracy, init_op, saver, tensorboard_dir, x, y, versions, 10)
if __name__ == "__main__":
    main()
