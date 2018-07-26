#!/usr/bin/env python3
# Copyright 2017 Lukas Schmelzeisen. All Rights Reserved.
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

from lib.config import REPLAYS_PARSED_DIR
from data import simulation_pb2


def model_frank(xs_train, xs_test, ys_train, ys_test, input_shape, num_classes, depth):
    # Set up model architecture.
    model = Sequential()
    model.add(Conv3D(32, kernel_size=[depth,4,4], activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1,4,4)))
    model.add(Conv3D(64, kernel_size=[1,4,4], activation='relu'))
    model.add(MaxPooling3D(pool_size=(1,2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Print summary of model architecture and parameters.
    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy'])

    # Fit model to training data.
    model.fit(
        xs_train,
        ys_train,
        epochs=10,
        batch_size=50,
        validation_split=1 / 9,
        verbose=2)

        # Evaluate model on testing data.
    score = model.evaluate(
        xs_test,
        ys_test,
        batch_size=50)
    for metric_name, metric_value in zip(model.metrics_names, score):
        print('{} = {}'.format(metric_name.capitalize(), metric_value))

    print('Done.', file=sys.stderr)