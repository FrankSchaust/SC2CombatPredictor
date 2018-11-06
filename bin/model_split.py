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
from lib.config import REPLAYS_PARSED_DIR, REPO_DIR
from data import simulation_pb2
from bin.load_batch import load_batch
from bin.modules import *

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    i = 0
    learning_rates = [0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    ### constant declarations, the different architectures will be iterated by chosing different learning rates and ratios of convolutions to fully connected layers
    epochs = 30
    batch_size = 20
    capped_batch = 100
    num_classes = 3
    depth = 13
    
	# Loading example files
    replay_parsed_files = []
    replay_parsed_files = build_file_array(version=['1_3c'])
    
    for lr in learning_rates: 
        # build the folder structure for tensorboard logs
        base_dir = os.path.join(REPO_DIR, 'tensorboard_logs', 'Model_Split', 'LearningRate_'+str(lr)+'_Sample_Size_'+str(capped_batch))
        os.makedirs(base_dir, exist_ok=True)
        sub_dirs = get_immediate_subdirectories(base_dir)
        last_run_fin = get_number_of_last_run(base_dir, sub_dirs)
        # every structure will be trained 10 times
        for n in range(1):
            last_run_fin += 1
            tensorboard_dir = os.path.join(base_dir, 'Run '+str(last_run_fin))
            os.makedirs(tensorboard_dir, exist_ok=True)
            print('Model and Logs saved at %s' % (tensorboard_dir))
            run_cnn(replays=replay_parsed_files, lr=lr, epochs=epochs, batch_size=batch_size, capped_batch=capped_batch, tensorboard_dir=tensorboard_dir)
 
def run_cnn(replays=[], lr=0.5, epochs=15, batch_size=10, capped_batch=100, depth=13, num_classes=3, tensorboard_dir=""): 
    acc = 0
    t_acc = 0
    x = tf.placeholder(tf.float32, shape=[None, depth, 84, 84, 1])
    y = tf.placeholder(tf.float32, shape=[None, num_classes])
    x_ = x
    ### player_relative
    with tf.name_scope('Player_Relative_Convolution'):
        x_rel = tf.slice(x_, [0, 0, 0, 0, 0], [-1, 1, 84, 84, 1])
        tf.reshape(x_rel, [-1, 1, 84, 84, 1])
        x_rel = tf.layers.conv3d(inputs=x_rel, filters=64, kernel_size=[1,3,3], strides=(1,1,1), kernel_regularizer=tf.keras.regularizers.l2(1e-4), padding='SAME', activation=tf.nn.relu)
        x_rel = tf.layers.max_pooling3d(inputs=x_rel, pool_size=[1,4,4], strides=(1,2,2))
        print_layer_details(tf.contrib.framework.get_name_scope(), x_rel.get_shape())
    ### unit dependend layers
    ### power, unit_type, hit_points, energy, shields
    with tf.name_scope('Unit_dependend_Convolution'):
        x1 = tf.slice(x_, [0, 6, 0, 0, 0], [-1, 1, 84, 84, 1])
        x2 = tf.slice(x_, [0, 8, 0, 0, 0], [-1, 1, 84, 84, 1])
        x3 = tf.slice(x_, [0, 9, 0, 0, 0], [-1, 1, 84, 84, 1])
        x4 = tf.slice(x_, [0, 10, 0, 0, 0], [-1, 1, 84, 84, 1])
        prep_layers = tf.concat([x1, x2, x3, x4], 1)
        tf.reshape(prep_layers, [-1, 4, 84, 84, 1])        
        x_up = tf.layers.conv3d(inputs=prep_layers, filters=64, kernel_size=[prep_layers.get_shape()[1],3,3], strides=(1,1,1), kernel_regularizer=tf.keras.regularizers.l2(1e-4), padding='SAME', activation=tf.nn.relu)
        x_up = tf.layers.max_pooling3d(inputs=x_up, pool_size=[1,4,4], strides=(1,2,2))
        print_layer_details(tf.contrib.framework.get_name_scope(), x_up.get_shape())
    ### player id
    with tf.name_scope('Player_ID_Convolution'):
        x_pid = tf.slice(x_, [0, 5, 0, 0, 0], [-1, 1, 84, 84, 1])
        tf.reshape(x_pid, [-1, 1, 84, 84, 1])
        x_pid = tf.layers.conv3d(inputs=x_pid, filters=64, kernel_size=[1,3,3], strides=(1,1,1), kernel_regularizer=tf.keras.regularizers.l2(1e-4), padding='SAME', activation=tf.nn.relu)
        x_pid = tf.layers.max_pooling3d(inputs=x_pid, pool_size=[1,4,4], strides=(1,2,2))
        print_layer_details(tf.contrib.framework.get_name_scope(), x_pid.get_shape())
    ### misc data 
    ### layers with mostly zeros as values are excluded
    with tf.name_scope('Misc_Values_Convolution'):
        x5 = tf.slice(x_, [0, 2, 0, 0, 0], [-1, 1, 84, 84, 1])
        x6 = tf.slice(x_, [0, 11, 0, 0, 0], [-1, 1, 84, 84, 1])
        misc = tf.concat([x5, x6], 1)
        tf.reshape(misc, [-1, 2, 84, 84, 1])
        x_misc = tf.layers.conv3d(inputs=misc, filters=64, kernel_size=[misc.get_shape()[1],3,3], strides=(1,1,1), kernel_regularizer=tf.keras.regularizers.l2(1e-4), padding='SAME', activation=tf.nn.relu)
        x_misc = tf.layers.max_pooling3d(inputs=x_misc, pool_size=[1,4,4], strides=(1,2,2))
        print_layer_details(tf.contrib.framework.get_name_scope(), x_misc.get_shape())
    with tf.name_scope('Concatenation'):
        x_conc = tf.concat([x_rel, x_up, x_pid, x_misc], 1)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_conc.get_shape())
    with tf.name_scope('general_conv_A'):
        x_cnvd = tf.layers.conv3d(inputs=x_conc, filters=128, kernel_size=[x.get_shape()[1], 5, 5], strides=(1,1,1), kernel_regularizer=tf.keras.regularizers.l2(1e-4), padding='SAME', activation=tf.nn.relu)
        x_maxpool2 = tf.layers.max_pooling3d(inputs=x_cnvd, pool_size=[1,4,4], strides=(1,2,2))
        print_layer_details(tf.contrib.framework.get_name_scope(), x_maxpool2.get_shape())
    with tf.name_scope('general_conv_B'):
        x_cnvd2 = tf.layers.conv3d(inputs=x_maxpool2, filters=256, kernel_size=[x_maxpool2.get_shape()[1], 5, 5], strides=(1,1,1), kernel_regularizer=tf.keras.regularizers.l2(1e-4), padding='SAME', activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_cnvd2.get_shape())
    with tf.name_scope('Average_and_Flatten_inputs'):
        x_avg = tf.layers.max_pooling3d(inputs=x_cnvd2, pool_size=[1,x_cnvd2.get_shape()[2]//2 ,x_cnvd2.get_shape()[3]//2], strides=(1,x_cnvd2.get_shape()[2]//2,x_cnvd2.get_shape()[3]//2))
        x_flat = tf.layers.flatten(x_avg)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_flat.get_shape())
    with tf.name_scope('Dense_Layer_A'):    
        x_dense = tf.layers.dense(inputs=x_flat, units=128, activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_dense.get_shape())
    with tf.name_scope('Final_Layer'):
        y_ = tf.layers.dense(inputs=x_dense, units=num_classes, activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), y_.get_shape())
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
        close_matchups, supplies = filter_close_matchups(replays, supply_limit=5)

        train_file_indices, test_file_indices = generate_random_indices(file_count=len(close_matchups), cap=cap, split_ratio=0.9) 
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
                # _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                test_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                #sys.stdout.write("\r[%-20s] %.2f%% --- Batch %d from %d" % ('='*int(((i+1)/total_batch)*20), ((i+1)/total_batch)*100, i+1, total_batch))
                #sys.stdout.flush()
                #avg_cost += c / total_batch
                #xs_test.append(batch_x)
                #ys_test.append(batch_y)
                t_acc += test_acc
                print(" --- Result of Epoch:", (epoch + 1), "Train accuracy: {:.2f}".format(acc*100), "% cost: {:.3f}".format(avg_cost), " test accuracy on {:d}".format(len(test_file_indices)), "samples: {:.2f}".format(t_acc*100), "%")
            else:
                batches = int(len(test_file_indices)/batch_size)
                for i in range(batches):
                    batch_x, batch_y, lis = load_batch(replays, indices=test_file_indices, capped_batch=batch_size, run=i, lastindex=lis,)
                    # _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
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
        for i in range(1000):
            xs, ys, li = load_batch(replays, indices=remaining_indices, capped_batch=1, run=i, lastindex=li)
            acc = sess.run(accuracy, feed_dict={x: xs, y: ys})
            #print(remaining_supplies[i])
            supply_acc[int(remaining_supplies[i]*2)] += acc
            supply_count[int(remaining_supplies[i]*2)] += 1
            if i%100==0 and i>0:
                print("%4d samples evaluated." % (i))
        for i in range(10):
            test_summary = tf.Summary(value=[tf.Summary.Value(tag='accuracy by supplies', simple_value=(supply_acc[i]/supply_count[i]))])
            summary_writer.add_summary(summary=test_summary, global_step=i)
            print("Accuracy for samples with a supply difference of %.1f: %6.2f%%" % (i/2, (supply_acc[i]/supply_count[i])))
        print("Overall accuracy on %5d samples: %6.2f%%" % (len(remaining_indices), sum(supply_acc)/sum(supply_count)))
        #writer.add_graph(sess.graph)
        #print(sess.run(accuracy, feed_dict={x: xs_test, y: ys_test}))
if __name__ == "__main__":
    main()
