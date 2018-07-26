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
from keras.optimizers import Adadelta
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, Dropout, MaxPooling2D, Conv3D, MaxPooling3D, BatchNormalization
from keras import metrics
from absl import app
from bin.load_batch import load_batch

from lib.config import REPLAYS_PARSED_DIR
from data import simulation_pb2


def model_reduced_comp(input_shape, num_classes, depth, replays, epochs):
    # Set up model architecture.
    model = Sequential()
    model.add(Conv3D(64, kernel_size=[depth,4,4], padding='same', activation='relu',
                     input_shape=input_shape))
    model.add(Conv3D(64, kernel_size=[1,4,4], strides=(1,4,4), activation='relu'))
    model.add(Conv3D(128, kernel_size=[depth,4,4], padding='same', activation='relu'))
    model.add(Conv3D(128, kernel_size=[1,4,4], strides=(1,4,4), activation='relu'))
    model.add(Conv3D(256, kernel_size=[depth,1,1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.75))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Print summary of model architecture and parameters.
    model.summary()
    #adadelta = Adadelta(lr=0.00001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=[metrics.categorical_accuracy])
    xs_train = []
    xs_test = []
    ys_train = []
    ys_test = []
    # Fit model to training data.
    xs_train, xs_test, ys_train, ys_test = load_batch(replays, 5000, 0, True)#Array with filenames, count of replay files to load, arg, training set?
    #print(xs_test)
    tbcallback = keras.callbacks.TensorBoard(log_dir='./', histogram_freq=1)
    model.fit(
                xs_train,
                ys_train,
                epochs=30,
                batch_size=25,
                validation_split=1 / 9,
                verbose=1,
                callbacks=[tbcallback]
                )
    
        # Evaluate model on testing data.
    score = model.evaluate(
        xs_test,
        ys_test,
        batch_size=50)
    for metric_name, metric_value in zip(model.metrics_names, score):
        print('{} = {}'.format(metric_name.capitalize(), metric_value))

    print('Done.', file=sys.stderr)