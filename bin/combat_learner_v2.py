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
from bin.model_lukas_3D import model_lukas
from bin.model_frank_3D import model_frank
from bin.model_reduced_computation_3D import model_reduced_comp
from bin.load_batch import load_batch

import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, Dropout, MaxPooling2D, Conv3D, MaxPooling3D
from absl import app

from lib.config import REPLAYS_PARSED_DIR
from data import simulation_pb2


def main(unused_argv):
    # Set numpy print options so that numpy arrays containing feature layers
    # are printed completely. Useful for debugging.
    np.set_printoptions(threshold=(84 * 84), linewidth=(84 * 2 + 10))

    replay_parsed_files = []
    for root, dir, files in os.walk(REPLAYS_PARSED_DIR):
        for file in files:
            if file.endswith(".SC2Replay_parsed.gz"):
                replay_parsed_files.append(os.path.join(root, file))
	
    #shaping configure
    depth = 4
    img_rows, img_cols = 84, 84
    num_classes = 3
    input_shape = (depth, img_rows,img_cols,1)

	
    #model_lukas(xs_train, xs_test, ys_train, ys_test, input_shape, num_classes)
    #model_frank(xs_train, xs_test, ys_train, ys_test, input_shape, num_classes, 3)
    model_reduced_comp(input_shape, num_classes, depth, replay_parsed_files, 15)
	
	

def entry_point():
    app.run(main)


if __name__ == '__main__':
    app.run(main)
