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
    learning_rates = [0.05]
    ### constant declarations, the different architectures will be iterated by chosing different learning rates and ratios of convolutions to fully connected layers
    conv_to_fc_ratio = [0.1]
    epochs = 20
    batch_size = 10
    capped_batch = 50
    num_classes = 3
    depth = 13
    r=1
    
	# Loading example files
    replay_parsed_files = []
    replay_parsed_files = build_file_array(version=['1_3b'])
    print(learning_rates, conv_to_fc_ratio)
    
    for lr in learning_rates:
        for cfr in conv_to_fc_ratio:
            # build the folder structure for tensorboard logs
            
            # every structure will be trained 10 times
            for n in range(1):
                tensorboard_dir = os.path.join(REPO_DIR, 'tensorboard_logs', 'inception', 'LearningRate_'+str(lr)+'_ConvToFcRatio_'+str(cfr)+'_SampleSize_'+str(capped_batch), 'Run '+str(r+n))
                os.makedirs(tensorboard_dir, exist_ok=True)
                run_cnn(replays=replay_parsed_files, lr=lr, cfr=cfr, epochs=epochs, capped_batch=capped_batch, tensorboard_dir=tensorboard_dir)
                
                
def print_layer_details(name_scope, shape):
    print("Layer: %-20s --- Output Dimension: %-25s" % (name_scope, shape))
   
def run_cnn(replays=[], lr=0.5, cfr=1, epochs=15, batch_size=10, capped_batch=100, depth=13, num_classes=3, tensorboard_dir=""):
    acc = 0
    t_acc = 0
    x = tf.placeholder(tf.float32, shape=[None, depth, 84, 84, 1])
    y = tf.placeholder(tf.float32, shape=[None, num_classes])
    
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
        x_ = tf.layers.conv3d(inputs=x, filters=96, kernel_size=[1, 3, 3], strides=(1,1,1), padding='same', activation=tf.nn.relu)
        x_ = tf.layers.conv3d(inputs=x_, filters=96, kernel_size=[1, 3, 3], strides=(1,1,1), padding='same', activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
        x_ = tf.layers.batch_normalization(inputs=x_, training=True)
    with tf.name_scope("Layer_B"):
        x_ = tf.layers.conv3d(inputs=x_, filters=96, kernel_size=[1, 3, 3], strides=(1,2,2), activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
    with tf.name_scope("Layer_C"):
        x_ = tf.layers.conv3d(inputs=x_, filters=192, kernel_size=[1, 3, 3], strides=(1,1,1), padding='same', activation=tf.nn.relu)
        x_ = tf.layers.conv3d(inputs=x_, filters=192, kernel_size=[1, 3, 3], strides=(1,1,1), padding='same', activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
        x_ = tf.layers.batch_normalization(inputs=x_, training=True)
    with tf.name_scope("Layer_D"):
        x_ = tf.layers.conv3d(inputs=x_, filters=192, kernel_size=[1, 3, 3], strides=(1,2,2), activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
    with tf.name_scope("Layer_E"):
        x_ = tf.layers.conv3d(inputs=x_, filters=192, kernel_size=[1, 3, 3], strides=(1,1,1), activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
    with tf.name_scope("Layer_F"):
        x_ = tf.layers.conv3d(inputs=x_, filters=192, kernel_size=[1, 1, 1], strides=(1,1,1), activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
    with tf.name_scope("Layer_G"):
        x_ = tf.layers.conv3d(inputs=x_, filters=10, kernel_size=[1, 1, 1], strides=(1,1,1), activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
    with tf.name_scope("Final_Layer"):
        x_avg = tf.layers.average_pooling3d(x_, pool_size=[13, 18, 18], strides=(1,1,1))
        x_flat = tf.layers.flatten(inputs=x_avg)
        # x_dense = tf.layers.dense(inputs=x_flat, units=16*int(9*(1-(cfr*(10/9)))))
        print_layer_details(tf.contrib.framework.get_name_scope(), x_flat.get_shape())
        y_ = tf.layers.dense(inputs=x_flat, units=num_classes)

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
        close_matchups, supplies = filter_close_matchups(replays, supply_limit=6)

        train_file_indices, test_file_indices = generate_random_indices(file_count=len(close_matchups), cap=cap, split_ratio=0.9) 
        print(len(train_file_indices))
        print(len(test_file_indices))
        print(train_file_indices, test_file_indices)
        
        remaining_indices, remaining_supplies = get_remaining_indices(file_count=len(close_matchups), ind1=train_file_indices, ind2=test_file_indices, supply = supplies)
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
                batch_x, batch_y, li = load_batch(replays, indices=train_file_indices, capped_batch=batch_size, run=i, lastindex=li, train=True)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                train_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                sys.stdout.write("\r[%-20s] %6.2f%% --- Batch %2d from %d --- Latest Acc: %6.2f%%" % ('='*int(((i+1)/batches)*20), ((i+1)/batches)*100, i+1, batches, train_acc*100))
                sys.stdout.flush()
                avg_cost += c / batches
                acc += train_acc / batches
            if len(test_file_indices) < 30: 
                batch_x, batch_y, lis = load_batch(replays, indices=test_file_indices, capped_batch=len(test_file_indices), run=1, lastindex=lis,)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                test_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                #sys.stdout.write("\r[%-20s] %.2f%% --- Batch %d from %d" % ('='*int(((i+1)/total_batch)*20), ((i+1)/total_batch)*100, i+1, total_batch))
                #sys.stdout.flush()
                #avg_cost += c / total_batch
                #xs_test.append(batch_x)
                #ys_test.append(batch_y)
                t_acc += test_acc
                print(" --- Result of Epoch:", (epoch + 1), "Train accuracy: {:.2f}".format(acc*100), "% cost: {:.3f}".format(avg_cost), " test accuracy on {:d}".format(len(test_file_indices)), "samples: {:.2f}".format(t_acc*100), "%")
            else:
                batches = int(len(test_file_indices))/batch_size
                for i in range(batches):
                    batch_x, batch_y, lis = load_batch(replays, indices=test_file_indices, capped_batch=batch_size, run=i, lastindex=lis,)
                    _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                    test_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                    #sys.stdout.write("\r[%-20s] %.2f%% --- Batch %d from %d" % ('='*int(((i+1)/total_batch)*20), ((i+1)/total_batch)*100, i+1, total_batch))
                    #sys.stdout.flush()
                    #avg_cost += c / total_batch
                    #xs_test.append(batch_x)
                    #ys_test.append(batch_y)
                    t_acc += test_acc / batches
                print(" --- Result of Epoch:", (epoch + 1), "Train accuracy: {:.2f}".format(acc*100), "% cost: {:.3f}".format(avg_cost), " test accuracy on {:d}".format(batches*batch_size), "samples: {:.2f}".format(t_acc*100), "%")
            train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=avg_cost),
                                              tf.Summary.Value(tag='train_accuracy', simple_value=acc),
                                              tf.Summary.Value(tag='test_accuracy', simple_value=t_acc)])



            summary_writer.add_summary(summary=train_summary, global_step=epoch)
            summary_writer.flush()


        print("\nTraining complete!")
        save_path = saver.save(sess, os.path.join(tensorboard_dir, "model.ckpt"))
        # Declare variables for the summary
        li = 0
        supply_acc = np.zeros(10)
        supply_count = np.zeros(10)
        for i in range(len(remaining_indices)):
            xs, ys, li = load_batch(replays, indices=remaining_indices, capped_batch=1, run=i, lastindex=li)
            acc = sess.run(accuracy, feed_dict={x: xs, y: ys})
            #print(remaining_supplies[i])
            supply_acc[int(remaining_supplies[i]*2)] += acc
            supply_count[int(remaining_supplies[i]*2)] += 1
            if i%100==0 and i>0:
                print("%4d samples evaluated." % (i))
        for i in range(10):
            test_summary = tf.Summary(value=[tf.Summary.Value(tag='acc'+str(i), simple_value=(supply_acc[i]/supply_count[i]))])
            summary_writer.add_summary(summary=test_summary, global_step=i)
            print("Accuracy for samples with a supply difference of %.1f: %6.2f%%" % (i/2, (supply_acc[i]/supply_count[i])))
        print("Overall accuracy on %5d samples: %6.2f%%" % (len(remaining_indices), sum(supply_acc)/sum(supply_count)))
        #writer.add_graph(sess.graph)
        #print(sess.run(accuracy, feed_dict={x: xs_test, y: ys_test}))
if __name__ == "__main__":
    main()
