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
import time
import gzip
import pprint

import numpy as np

from absl import app


from lib.config import REPLAYS_PARSED_DIR, REPO_DIR
from data import simulation_pb2
from bin.load_batch import load_batch
from bin.read_csv import read_csv

def main():
    # Set numpy print options so that numpy arrays containing feature layers
    # are printed completely. Useful for debugging.
    np.set_printoptions(threshold=(84 * 84), linewidth=(84 * 2 + 10))
    depth = 20
	# Loading example files
    replay_parsed_files = []
    print("Creating list of used files")
    for root, dir, files in os.walk(REPLAYS_PARSED_DIR):
        for file in files:
            if file.endswith(".SC2Replay_parsed.gz"):
                replay_parsed_files.append(os.path.join(root, file))
    print("Available Files: ", len(replay_parsed_files))
    li = 0
    emptyCounter = 0
    overallCounter = 0
    countPerFeature = np.zeros(21)
    while  li < len(replay_parsed_files):
        xs_t, xs_tt, ys_t, ys_tt, li = load_batch(replay_parsed_files, 1000, 0, li, True, False)
        xs_t = np.append(xs_t, xs_tt, axis=0)
        for test in xs_t:
            n = 0
            countPerFeature[20] += 1 
            for f in test:
                sum = np.sum(f, axis=1)
               # print(len(sum), sum)
                assert len(sum)==84
                left = np.sum(sum[:42], axis=0)
                right = np.sum(sum[42:], axis=0)
                sumum = np.sum(sum, axis=0)
            # #print(left, right)
                if sumum == 0:
                    #print("both empty")
                #if left == 0 or right == 0:
                    #print("one side empty!")
                    #emptyCounter += 1
                    countPerFeature[n] += 1
                n += 1
                
    print(countPerFeature) 
    
    replay_log_files = []
    match_arr = []
    logfailureCounter = 0
    for root, dir, files in os.walk(os.path.join(REPO_DIR, 'log')):
        for file in files:
            if file.endswith(".csv"):
                replay_log_files.append(os.path.join(root, file))
    i = 0
    print('Looking over', len(replay_log_files), 'files')
    while i < len(replay_log_files):
        match_arr.append(read_csv(replay_log_files[i]))
        i = i + 1
        #if i%10000 == 0:
            #print(i, 'csv-files loaded')    
            
    for match in match_arr:
        if not match['team_A'] or not match['team_B']:
           logfailureCounter += 1
    print(len(match_arr), logfailureCounter)
           
if __name__ == "__main__":
    main()