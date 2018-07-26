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


def model_lukas(xs_train, xs_test, ys_train, ys_test, input_shape, num_classes):
    # Set up model architecture.
    model_lukas = Sequential()
    model_lukas.add(Conv3D(32, kernel_size=[None,4,4], activation='relu',
                     input_shape=input_shape))
    model_lukas.add(Conv3D(64, kernel_size=[1,4,4], activation='relu'))
    model_lukas.add(MaxPooling3D(pool_size=(1,4,4)))
    model_lukas.add(Dropout(0.25))
    model_lukas.add(Flatten())
    model_lukas.add(Dense(128, activation='relu'))
    model_lukas.add(Dropout(0.5))
    model_lukas.add(Dense(num_classes, activation='softmax'))

    # Print summary of model architecture and parameters.
    model_lukas.summary()

    model_lukas.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy'])

    # Fit model to training data.
    model_lukas.fit(
        xs_train,
        ys_train,
        epochs=10,
        batch_size=50,
        validation_split=1 / 9,
        verbose=2)

        # Evaluate model on testing data.
    score = model_lukas.evaluate(
        xs_test,
        ys_test,
        batch_size=50)
    for metric_name, metric_value in zip(model.metrics_names, score):
        print('{} = {}'.format(metric_name.capitalize(), metric_value))

    print('Done.', file=sys.stderr)