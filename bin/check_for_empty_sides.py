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

import pandas as pd 
import numpy as np
from absl import app
import matplotlib.pyplot as plt
import matplotlib.ticker as mplt
from matplotlib.ticker import MaxNLocator

from bin.util import *
from lib.config import REPLAYS_PARSED_DIR, REPO_DIR
from data import simulation_pb2
from bin.load_batch import load_batch
from bin.hard_coded_decider import return_unit_values_by_id
from lib.unit_constants import *

def main():
    # Set numpy print options so that numpy arrays containing feature layers
    # are printed completely. Useful for debugging.
    np.set_printoptions(threshold=(84 * 84), linewidth=(84 * 2 + 10))
    depth = 13
    use_version = ['1_3a', '1_3b', '1_3c', '1_3d']
	# Loading example files
    replay_parsed_files = build_file_array(version=use_version)
    li = 0
    m = 0
    emptyCounter = 0
    overallCounter = 0
    countPerFeatureLeft = np.zeros(20)
    countPerFeatureRight = np.zeros(20)
    countPerFeatureMedianRight = np.zeros(20)
    countPerFeatureMedianLeft = np.zeros(20)
    firstentry = True
    rightc = []
    leftc = []
    while  li < len(replay_parsed_files):
        xs_t, ys_t, li = load_batch(replay_parsed_files, capped_batch=1000, run = m, lastindex=li)
        
        for test in xs_t:
            
            n = 0
            #countPerFeature[20] += 1 
            for f in test:
                sum = np.sum(f, axis=0)
                #if n == 8 and firstentry:
                    #print(sum)
                    #firstentry = False
               # print(len(sum), sum)
                assert len(sum)==84
                left = np.sum(np.abs(sum[:42]), axis=0)
                right = np.sum(np.abs(sum[42:]), axis=0)
                sumum = np.sum(sum, axis=0)
                if n == 8:
                    leftc.append(left)
                    rightc.append(right)
                    firstentry = False
            # #print(left, right)
                #if sumum == 0:
                    #print("both empty")
                if left == 0:
                    #print("one side empty!")
                    #emptyCounter += 1
                    countPerFeatureLeft[n] += 1
                    if n == 8:
                        print(replay_parsed_files[overallCounter])
                countPerFeatureMedianRight[n] +=  right
                if right == 0:
                    countPerFeatureRight[n] += 1
                    if n == 8:
                        print(replay_parsed_files[overallCounter])
                countPerFeatureMedianLeft[n] += left
                
                n += 1
            overallCounter += 1
        m += 1
    print(countPerFeatureLeft)
    print(countPerFeatureRight)
    print(countPerFeatureMedianLeft)
    print(countPerFeatureMedianRight)
    #calc median
    i = 0
    for o in countPerFeatureMedianLeft:
        countPerFeatureMedianLeft[i] = o / len(xs_t)# - countPerFeatureRight[i])
        i += 1
    i = 0 
    for m in countPerFeatureMedianRight:
        countPerFeatureMedianRight[i] = m / len(xs_t)# - countPerFeatureLeft[i])
        i += 1
     
    #print(countPerFeatureMedianLeft, countPerFeatureMedianRight)
    
    replay_log_files = []
    match_arr = []
    logfailureCounter = 0
    replay_log_files = build_file_array('logs', use_version)
    i = 0
    print('Looking over', len(replay_log_files), 'files')
    while i < len(replay_log_files):
        match_arr.append(read_csv(replay_log_files[i]))
        i = i + 1
    hpa = []
    hpb = []
    j = 0
    while j < len(match_arr):
        hp_A = 0
        hp_B = 0
        if len(match_arr[j]['team_A']) < 1 or len(match_arr[j]['team_B']) < 1:
           print(replay_log_files[j])
           logfailureCounter += 1
           j += 1
           continue
        for id in match_arr[j]['team_A']:
            if str(id) == '85':
                continue
            unit = return_unit_values_by_id(int(id))
            try:
                hp_A += unit['hp']
            except:
                print(id)
                print(unit)
        for id in match_arr[j]['team_B']:
            if str(id) == '85':
                continue
            unit = return_unit_values_by_id(int(id))
            hp_B += unit['hp'] 
        hpa.append(hp_A)
        hpb.append(hp_B)
        j += 1
    i = 0
    while i < len(hpa):
        if i % 1000 == 0:
            print(leftc[i], hpa[i], rightc[i], hpb[i]) 
        i += 1
    print(len(match_arr), logfailureCounter)
           
    names = ("Player Relative", "Height Map", "Visibility", "Creep", "Power", "Player ID", "Unit Type", "Selected", "Hit Points", "Energy", "Shields", "Unit Density", "Unit Density AA", "Mini Height Map", "Mini Visibility", "Mini Creep", "Mini Camera", "Mini Player ID", "Mini Player Relative", "Mini Selected")
    fig, ax = plt.subplots()
    
    index = np.arange(20)
    bar_width = 0.3
    
    opacity = 0.4
    error_config = {'ecocolor' : '0.3'}

    
    rects1 = ax.bar(index, countPerFeatureLeft, bar_width, alpha=opacity, color='b', error_kw=error_config, label='Left Side')    
    rects2 = ax.bar(index+bar_width, countPerFeatureRight, bar_width, alpha=opacity, color='r', error_kw=error_config, label='Right Side')
    ax.set_xlabel('Feature Maps')
    ax.set_ylabel('Count')
    ax.set_title('Count of empty sides per feature map')
    ax.set_xticks(index+bar_width/2)
    ax.set_xticklabels(names)
    ax.legend()
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1,
                    '%d' % int(height),
                    ha='center', va='bottom', rotation=90)
    autolabel(rects1)
    def autolabel1(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1,
                    '%d' % int(height),
                    ha='center', va='bottom', rotation=90)
    autolabel1(rects2)
    
    fig.tight_layout()
    plt.xticks(rotation=90)
    plt.show()
    
if __name__ == "__main__":
    main()