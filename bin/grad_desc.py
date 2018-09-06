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
from data import simulation_pb2

from bin.load_batch import load_batch
from bin.data_visualization import map_id_to_units_race


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.axes3d import Axes3D

from bin.util import *
from lib.unit_constants import *
from lib.config import SCREEN_RESOLUTION, MINIMAP_RESOLUTION, MAP_PATH, \
    REPLAYS_PARSED_DIR, REPLAY_DIR, REPO_DIR, STANDARD_VERSION
    
def main():
    learning_rates = np.arange(0.003, 0.005, 0.0005)
    training_epochs = 1000
    
    trackAcc = []
    trackAccs = []
    trackCost = []
    trackCosts = []
    for learning_rate in learning_rates:
        trackAcc, trackCost = run_grad_desc(learning_rate, training_epochs)
        trackAccs.append(trackAcc)
        trackCosts.append(trackCost)
    create_graphs(trackAccs, trackCosts, learning_rates, training_epochs)
def run_grad_desc(learning_rate=0.5, training_epochs = 10):
    # Graph Input
    x = tf.placeholder(tf.float32, [None, 94])
    y = tf.placeholder(tf.float32, [None, 3])
    
    # initialize weight and bias
    W = tf.Variable(tf.truncated_normal([94, 3]))
    
    # b = tf.Variable(tf.zeros([3]))
    
    # Construct Model
    logits = tf.matmul(x, W) #+ b
    
    pred = tf.nn.softmax(logits)
    # minimize error using cross entropy
    # cross_entropy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    init = tf.global_variables_initializer()
    
    trackAcc = []
    trackCost = []
    with tf.Session() as s: 
        s.run(init)
        xs_train, xs_test, ys_train, ys_test = load(version='1_3a')
        # loop to train for specified number of epochs
        for epoch in range(training_epochs):
            _, c = s.run([optimizer, cost], feed_dict={x: xs_train, y: ys_train})
            acc = s.run(accuracy, feed_dict={x: xs_test, y: ys_test})
            # track accuracy to display in graph when algorithm finished
            trackCost.append(c)
            trackAcc.append(acc*100)
            print('Epoch:', '%04d' % (epoch+1), "completed with an accuracy of:", "{:.3f}".format(acc), "cost=", "{:.9f}".format(c))
        # evaluate accuary when all training steps are completed
        print ("Accuracy:", accuracy.eval({x: xs_test, y: ys_test}))
        
        trackAcc = np.array(trackAcc)
        return trackAcc, trackCost
def create_graphs(trackAcc, trackCost, learning_rate, training_epochs=10):
        # create graph
        fig = plt.figure(figsize=plt.figaspect(4.))
        # add plot
        ax = fig.add_subplot(2,1,1)
        # create array that corresponds to the number of training steps as x-axis
        # y-axis is the accuracy in %
        a = np.arange(1, training_epochs+1)
        ax.set_title('Test Accuracy')
        i = 0
        for acc in trackAcc:
            ax.plot(a, acc, '-', label=learning_rate[i])
            i += 1
            
        bx = fig.add_subplot(2,1,2)
        bx.set_title('Cost by Epoch')
        i = 0
        for cost in trackCost:
            bx.plot(a, cost, '-', label=learning_rate[i])
            i += 1
        plt.show()

# function to load the csv-data and construct the input array as return
# input array is a vector with one entry per possible unit id 
# 94 entries 47 per combat party
def load(version = STANDARD_VERSION, file_version='single'):
    match_arr = []
    # load file(s) depending on desired input and version number 
    if file_version == 'multiple':
        replay_log_files = []
        

        replay_log_files = build_file_array('logs', version)
        i = 0
        print('Looking over', len(replay_log_files), 'files')
        while i < len(replay_log_files):
            match_arr.append(read_csv(replay_log_files[i]))
            i = i + 1
            if i%10000 == 0:
                print(i, 'csv-files loaded')

        print('match_arr built...')
    if file_version == 'single':
        file_path = os.path.join(REPO_DIR, 'all_csv_from_version_' + version + '.csv')
        match_arr = read_summed_up_csv(file_path, 250)
    unit_vector_A = np.zeros(47)
    unit_vector_B = np.zeros(47)
    xs = []
    ys = []
    #print(match_arr[0], match_arr[3])
    n=0
    typeerror = 0
    for match in match_arr:
    
        # if str(match['winner_code']) == str(2):
            # continue
        try:
            for id in match['team_A']:
                id = int(id.replace("'", ""))
                if id == 85:
                    continue
                if id == 9:
                    unit_vector_A[0] += 1
                    
                if id == 12 or id == 13 or id == 15 or id == 17:
                    unit_vector_A[1] += 1
                    
                if id == 104:
                    unit_vector_A[2] += 1
                    
                if id == 105:
                    unit_vector_A[3] += 1
                    
                if id == 106:
                    unit_vector_A[4] += 1
                    
                if id == 107:
                    unit_vector_A[5] += 1
                    
                if id == 108:
                    unit_vector_A[6] += 1
                    
                if id == 109:
                    unit_vector_A[7] += 1
                    
                if id == 110:
                    unit_vector_A[8] += 1
                    
                if id == 111:
                    unit_vector_A[9] += 1
                    
                if id == 112:
                    unit_vector_A[10] += 1
                    
                if id == 114:
                    unit_vector_A[11] += 1
                    
                if id == 126:
                    unit_vector_A[12] += 1
                    
                if id == 129:
                    unit_vector_A[13] += 1
                    
                if id == 289:
                    unit_vector_A[14] += 1
                    
                if id == 499:
                    unit_vector_A[15] += 1
                    
                if id == 4:
                    unit_vector_A[16] += 1
                    
                if id == 10:
                    unit_vector_A[17] += 1
                    
                if id == 73:
                    unit_vector_A[18] += 1
                    
                if id == 74:
                    unit_vector_A[19] += 1
                    
                if id == 75:
                    unit_vector_A[20] += 1
                    
                if id == 76:
                    unit_vector_A[21] += 1
                    
                if id == 77:
                    unit_vector_A[22] += 1
                    
                if id == 78:
                    unit_vector_A[23] += 1
                    
                if id == 79:
                    unit_vector_A[24] += 1
                    
                if id == 80:
                    unit_vector_A[25] += 1
                    
                if id == 82:
                    unit_vector_A[26] += 1
                    
                if id == 83:
                    unit_vector_A[27] += 1
                    
                if id == 84:
                    unit_vector_A[28] += 1
                    
                if id == 141:
                    unit_vector_A[29] += 1
                    
                if id == 311:
                    unit_vector_A[30] += 1
                    
                if id == 694:
                    unit_vector_A[31] += 1
                    
                if id == 32 or id == 33:
                    unit_vector_A[32] += 1
                    
                if id == 34 or id == 35:
                    unit_vector_A[33] += 1
                    
                if id == 45:
                    unit_vector_A[34] += 1
                    
                if id == 48:
                    unit_vector_A[35] += 1
                    
                if id == 49:
                    unit_vector_A[36] += 1
                    
                if id == 50:
                    unit_vector_A[37] += 1
                    
                if id == 51:
                    unit_vector_A[38] += 1
                    
                if id == 52:
                    unit_vector_A[39] += 1
                    
                if id == 53 or id == 484:
                    unit_vector_A[40] += 1
                    
                if id == 54:
                    unit_vector_A[41] += 1
                    
                if id == 55:
                    unit_vector_A[42] += 1
                    
                if id == 56:
                    unit_vector_A[43] += 1
                    
                if id == 57:
                    unit_vector_A[44] += 1
                    
                if id == 268:
                    unit_vector_A[45] += 1
                    
                if id == 692:
                    unit_vector_A[46] += 1
                    
            for id in match['team_B']:
                id = int(id.replace("'", ""))
                if id == 85:
                    continue
                    
                if id == 9:
                    unit_vector_B[0] += 1
                    
                if id == 12 or id == 13 or id == 15 or id == 17:
                    unit_vector_B[1] += 1
                    
                if id == 104:
                    unit_vector_B[2] += 1
                    
                if id == 105:
                    unit_vector_B[3] += 1
                    
                if id == 106:
                    unit_vector_B[4] += 1
                    
                if id == 107:
                    unit_vector_B[5] += 1
                    
                if id == 108:
                    unit_vector_B[6] += 1
                    
                if id == 109:
                    unit_vector_B[7] += 1
                    
                if id == 110:
                    unit_vector_B[8] += 1
                    
                if id == 111:
                    unit_vector_B[9] += 1
                    
                if id == 112:
                    unit_vector_B[10] += 1
                    
                if id == 114:
                    unit_vector_B[11] += 1
                    
                if id == 126:
                    unit_vector_B[12] += 1
                    
                if id == 129:
                    unit_vector_B[13] += 1
                    
                if id == 289:
                    unit_vector_B[14] += 1
                    
                if id == 499:
                    unit_vector_B[15] += 1
                    
                if id == 4:
                    unit_vector_B[16] += 1
                    
                if id == 10:
                    unit_vector_B[17] += 1
                    
                if id == 73:
                    unit_vector_B[18] += 1
                    
                if id == 74:
                    unit_vector_B[19] += 1
                    
                if id == 75:
                    unit_vector_B[20] += 1
                    
                if id == 76:
                    unit_vector_B[21] += 1
                    
                if id == 77:
                    unit_vector_B[22] += 1
                    
                if id == 78:
                    unit_vector_B[23] += 1
                    
                if id == 79:
                    unit_vector_B[24] += 1
                    
                if id == 80:
                    unit_vector_B[25] += 1
                    
                if id == 82:
                    unit_vector_B[26] += 1
                    
                if id == 83:
                    unit_vector_B[27] += 1
                    
                if id == 84:
                    unit_vector_B[28] += 1
                    
                if id == 141:
                    unit_vector_B[29] += 1
                    
                if id == 311:
                    unit_vector_B[30] += 1
                    
                if id == 694:
                    unit_vector_B[31] += 1
                    
                if id == 32 or id == 33:
                    unit_vector_B[32] += 1
                    
                if id == 34 or id == 35:
                    unit_vector_B[33] += 1
                    
                if id == 45:
                    unit_vector_B[34] += 1
                    
                if id == 48:
                    unit_vector_B[35] += 1
                    
                if id == 49:
                    unit_vector_B[36] += 1
                    
                if id == 50:
                    unit_vector_B[37] += 1
                    
                if id == 51:
                    unit_vector_B[38] += 1
                    
                if id == 52:
                    unit_vector_B[39] += 1
                    
                if id == 53 or id == 484:
                    unit_vector_B[40] += 1
                    
                if id == 54:
                    unit_vector_B[41] += 1
                    
                if id == 55:
                    unit_vector_B[42] += 1
                    
                if id == 56:
                    unit_vector_B[43] += 1
                    
                if id == 57:
                    unit_vector_B[44] += 1
                    
                if id == 268:
                    unit_vector_B[45] += 1
                    
                if id == 692:
                    unit_vector_B[46] += 1

            unit_vector = np.append(unit_vector_A, unit_vector_B) 
            xs.append(unit_vector)
            ys.append(int(match['winner_code']))
        except TypeError:
            print(id)
            typeerror += 1
            continue
        except ZeroDivisionError:
            continue  
    #print(typeerror)
    #print(xs[0])
    ys = keras.utils.to_categorical(ys, num_classes=3)
    
    split = int(len(xs)*0.1)
    # # Make train / test split
    xs_train = xs[:-split]
    ys_train = ys[:-split]
    xs_test = xs[-split:]
    ys_test = ys[-split:]
    return xs_train, xs_test, ys_train, ys_test
  
if __name__ == "__main__":
    main()

