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
from lib.config import REPLAYS_PARSED_DIR, REPO_DIR
from data import simulation_pb2
from bin.load_batch import load_batch

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    i = 0
    learning_rates = []
    ### constant declarations, the different architectures will be iterated by chosing different learning rates and ratios of convolutions to fully connected layers
    for i in range(5):
        learning_rates.append(0.5*(10**(-i)))
      
    conv_to_fc_ratio = np.arange(0, 1.1, 0.1)
    epochs = 30
    batch_size = 10
    capped_batch = 100
    num_classes = 3
    depth = 13
    
	# Loading example files
    replay_parsed_files = []
    replay_parsed_files = build_file_array(version='1_3a')
    print(learning_rates, conv_to_fc_ratio)
    
    for lr in learning_rates:
        for cfr in conv_to_fc_ratio:
            # build the folder structure for tensorboard logs
            tensorboard_dir = os.path.join(REPO_DIR, 'tensorboard_logs', 'inception', 'LearningRate_'+str(lr)+'_ConvToFcRatio_'+str(cfr)+'_SampleSize_'+str(capped_batch))
            os.makedirs(tensorboard_dir, exist_ok=True)
            # every structure will be trained 10 times
            for n in range(10):
                run_cnn(replays=replay_parsed_files, lr=lr, cfr=cfr, epochs=epochs, capped_batch=capped_batch, tensorboard_dir=tensorboard_dir)

def inception(inputs, kernels=[1,1,1,1,1,1]):
 ### TODO                
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
    inception_bn = tf.layers.batch_normalization(inputs=inception, training=True)
    return inception_bn
   
def run_cnn(replays=[], lr=0.5, cfr=1, epochs=15, batch_size=10, capped_batch=100, depth=13, num_classes=3, tensorboard_dir=""):
    acc = 0
    t_acc = 0
    x = tf.placeholder(tf.float32, shape=[None, depth, 84, 84, 1])
    y = tf.placeholder(tf.float32, shape=[None, num_classes])
    inception_kernels = [
        [64, 96, 128, 16, 32, 32],
        [128, 128, 192, 32, 96, 64],
        [192, 96, 208, 16, 48, 64],
        [160, 112, 224, 24, 64, 64],
        [128, 128, 256, 24, 64, 64],
        [112, 144, 288, 32, 64, 64],
        [256, 160, 320, 32, 128, 128],
        [256, 160, 320, 32, 128, 128],
        [384, 192, 384, 48, 128, 128]
        ]
    ### based on a depth of 10 layers we define the ratio between inceptions and fully-connected layers as the ratio between the count of layers
    ### the architectures based on the ratio should be defined by hand, as the kernel specifications vary depending on the depth of the inceptions
    ### fully connected layers may increase the number of kernel with greater depth. 
    ### could be interesting what diffences may occure when the kernel size for deep fcl structures remain constant
    
    # # # Layer A 
    # convoluitional layer with 6x6 kernel, 2 stride and 64 kernels
    # max pool is 3x3 with 2 stride
    # input  13 x 84 x 84 x 1
    # output 13 x 21 x 21 x 64
    with tf.name_scope("Layer_A"):
        conv1 = tf.layers.conv3d(inputs=x, filters=64, kernel_size=[1, 6, 6], strides=(1,1,1), padding='same', activation=tf.nn.relu)
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
        max_pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[1, 2, 2], strides=(1,2,2))
        x_ = tf.layers.batch_normalization(inputs=max_pool2, training=True)
        
    for i in range(int(9*cfr)):
        with tf.name_scope("Layer_C_"+str(i)):
            x_ = inception(inputs=x_, kernels=inception_kernels[i])
            if i == 1 or i == 6 or i == 8:
                x_ = tf.layers.max_pooling3d(inputs=x_, pool_size=[1, 2, 2], strides=(1,2,2))
    with tf.name_scope("Layer_D"):
        x_avg = tf.layers.average_pooling3d(x_, pool_size=[13, 1, 1], strides=(1,1,1))
        # print(avg_pool.get_shape())
        x_flat = tf.layers.flatten(inputs=x_avg)
        x_dense = tf.layers.dense(inputs=x_flat, units=16*int(9*(1-cfr)))
        y_ = tf.layers.dense(inputs=x_dense, units=num_classes)

    softmax = tf.nn.softmax(y_)
    
    
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
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
        # initialise the variables
        sess.run(init_op)
        # setup recording variables
        # add a summary to store the accuracy
        cap = capped_batch
        close_matchups_and_supply = filter_close_matchups(replays)
        
        close_matchups = close_matchups_and_supply['matches']
        #supplies = close_matchups_and_supply['supply']
        
        train_file_indices, test_file_indices = generate_random_indices(file_count=len(close_matchups), cap=cap, split_ratio=0.9) 
        print(len(train_file_indices))
        print(len(test_file_indices))
        print(train_file_indices, test_file_indices)
        
        #remaining_indices, remaining_supplies = get_remaining_indices(file_count=len(close_matchups), ind1=train_file_indices, ind2=test_file_indices, supply=supplies)
        for epoch in range(epochs):
            avg_cost = 0
            acc = 0
            t_acc = 0
            ys_test = []
            xs_test = []
            li = 0
            lis = 0
            last_batch_acc = 0
            batches = int(len(train_file_indices)/batch_size)
            for i in range(batches):
                batch_x, batch_y, li = load_batch(replays, train_indices=train_file_indices, capped_batch=batch_size, run=i, lastindex=li, train=True)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                train_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                sys.stdout.write("\r[%-20s] %6.2f%% --- Batch %2d from %d --- Latest Acc: %6.2f%%" % ('='*int(((i+1)/batches)*20), ((i+1)/batches)*100, i+1, batches, train_acc*100))
                sys.stdout.flush()
                avg_cost += c / batches
                acc += train_acc / batches
            if len(test_file_indices) < 30: 
                batch_x, batch_y, lis = load_batch(replays, test_indices=test_file_indices, capped_batch=len(test_file_indices), run=1, lastindex=lis,)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                test_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                #sys.stdout.write("\r[%-20s] %.2f%% --- Batch %d from %d" % ('='*int(((i+1)/total_batch)*20), ((i+1)/total_batch)*100, i+1, total_batch))
                #sys.stdout.flush()
                #avg_cost += c / total_batch
                #xs_test.append(batch_x)
                #ys_test.append(batch_y)
                t_acc += test_acc
                print(" --- Result of Epoch:", (epoch + 1), "Train accuracy: {:.3f}".format(acc), "cost: {:.3f}".format(avg_cost), " test accuracy on {:d}".format(len(test_file_indices)), "samples: {:.3f}".format(t_acc))
            else:
                batches = int(len(test_file_indices))/batch_size
                for i in range(batches):
                    batch_x, batch_y, lis = load_batch(replays, test_indices=test_file_indices, capped_batch=batch_size, run=i, lastindex=lis,)
                    _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                    test_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                    #sys.stdout.write("\r[%-20s] %.2f%% --- Batch %d from %d" % ('='*int(((i+1)/total_batch)*20), ((i+1)/total_batch)*100, i+1, total_batch))
                    #sys.stdout.flush()
                    #avg_cost += c / total_batch
                    #xs_test.append(batch_x)
                    #ys_test.append(batch_y)
                    t_acc += test_acc / batches
                print(" --- Result of Epoch:", (epoch + 1), "Train accuracy: {:.3f}".format(acc), "cost: {:.3f}".format(avg_cost), " test accuracy on {:d}".format(batches*batch_size), "samples: {:.3f}".format(t_acc))
            train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=avg_cost),
                                              tf.Summary.Value(tag='train_accuracy', simple_value=acc),
                                              tf.Summary.Value(tag='test_accuracy', simple_value=t_acc)])



            summary_writer.add_summary(summary=train_summary, global_step=epoch)
            summary_writer.flush()


        print("\nTraining complete!")
        save_path = saver.save(sess, os.path.join(tensorboard_dir, "model.ckpt")
        #writer.add_graph(sess.graph)
        #print(sess.run(accuracy, feed_dict={x: xs_test, y: ys_test}))
if __name__ == "__main__":
    main()