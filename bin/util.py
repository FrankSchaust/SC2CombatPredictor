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
import gzip
import time
import csv
import random
import six
import pandas as pd
import tensorflow as tf
from absl import app

from lib.unit_constants import *
from lib.config import SCREEN_RESOLUTION, MINIMAP_RESOLUTION, MAP_PATH, \
    REPLAYS_PARSED_DIR, REPLAY_DIR, REPO_DIR, STANDARD_VERSION



def get_number_of_last_run(base_dir, sub_dirs):
    last_run_fin = 0
    for sub_dir in sub_dirs:
        run_dir = os.path.relpath(sub_dir, base_dir)
        run_ind = run_dir.split(' ')[1]
        print(run_ind, last_run_fin)
        if int(run_ind) > int(last_run_fin):
            last_run_fin = int(run_ind)
    return int(last_run_fin)


def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, name))]

def get_params(tv):
    para = 0
    for v in tv:
        shape = v.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        para += variable_parameters
    return para

def print_layer_details(name_scope, shape):
    para = get_params(tf.trainable_variables())
    print("Layer: %-50s --- Output Dimension: %-30s --- Sum of Trainable Parameters: %-10s" % (name_scope, shape, para))

# Input arguments: 
# type = String with associated files to load
#       Either: 'replays' to build a file_array containing the parsed_replays names
#       Or: 'logs' to build a file_array containing the logs names
# version = String to define the version of the data; refers to the suffix of the folders specified 'log_v1_3' is specified by '_v1_3'
# Return: Array with all files in the given directory 
def build_file_array(type = 'replays', version = [STANDARD_VERSION]):
    file_array = []
    DIRECTORY = []
    PATH = []
    # Routine for replay data
    if type == 'replays':
        if version == []:
            DIRECTORY.append(os.path.join(REPLAYS_PARSED_DIR))
        else:
            for v in version:
                DIRECTORY.append(os.path.join(REPLAYS_PARSED_DIR, v))
        print("Creating list of used files")
        for d in DIRECTORY:
            for root, dir, files in os.walk(d):
                for file in files:
                    if file.endswith(".gz"):
                        if file.endswith("_2.SC2Replay_parsed.gz") or file.endswith("_1.SC2Replay_parsed.gz"):
                            continue
                        file_array.append(os.path.join(root, file))
        print("Available Files: ", len(file_array))
    # Routine for log data    
    if type == 'logs':
        if version == []:
            PATH.append(os.path.join(REPO_DIR, 'log'))
        else:
            for v in version:
                PATH.append(os.path.join(REPO_DIR, 'log', v))
        for p in PATH:
            for root, dir, files in os.walk(p):
                for file in files:
                    if file.endswith(".csv"):
                        file_array.append(os.path.join(root, file))
    if type == 'csv':
        for v in version:
            PATH.append(os.path.join(REPO_DIR, 'Proc_Data_', v))
        for p in PATH:
            for root, dir, files in os.walk(p):
                for file in files:
                    if file.endswith(".csv"):
                        file_array.append(os.path.join(root, file))
    print("Available Files: ", len(file_array))
    return file_array
    
def generate_random_indices(file_count = 0, cap = 0, split_ratio = 0.9):
    file_array = random.sample(range(0, file_count-1), cap)
    split = int(len(file_array) * (1-split_ratio))
    train_file_array = file_array[:-(split+1)]
    test_file_array = file_array[-(split+1):]
    
    return train_file_array, test_file_array


def remove_zero_layers(x_):
    # remove zero matrices
    x0 = tf.slice(x_, [0, 0, 0, 0, 0], [-1, 1, 84, 84, 1])
    x1 = tf.slice(x_, [0, 6, 0, 0, 0], [-1, 1, 84, 84, 1])
    x2 = tf.slice(x_, [0, 8, 0, 0, 0], [-1, 1, 84, 84, 1])
    x3 = tf.slice(x_, [0, 9, 0, 0, 0], [-1, 1, 84, 84, 1])
    x4 = tf.slice(x_, [0, 10, 0, 0, 0], [-1, 1, 84, 84, 1])
    x5 = tf.slice(x_, [0, 5, 0, 0, 0], [-1, 1, 84, 84, 1])
    x6 = tf.slice(x_, [0, 2, 0, 0, 0], [-1, 1, 84, 84, 1])
    x7 = tf.slice(x_, [0, 11, 0, 0, 0], [-1, 1, 84, 84, 1])
    prep_layers = tf.concat([x0, x1, x2, x3, x4, x5, x6, x7], 1)
    tf.reshape(prep_layers, [-1, 8, 84, 84, 1])
    return prep_layers
        
def filter_close_matchups(replays = [], supply_limit = 0, versions = [STANDARD_VERSION], type='rep'):

    close_matchups = []
    supply_diff = []
    #for v in versions:
    v = versions[0]
    for replay in replays:
        if type == 'rep':
            LOG_SINGLE = os.path.join(REPO_DIR, 'log', v, os.path.relpath(replay, os.path.join(REPLAYS_PARSED_DIR, v)).replace('.SC2Replay_parsed.gz', '.csv'))
        if type == 'csv':
            LOG_SINGLE = os.path.join(REPO_DIR, 'log', v, os.path.relpath(replay, os.path.join(REPO_DIR, 'Proc_Data_', v)).replace('\\Layer.csv', '.csv'))
        if type == 'log':
            LOG_SINGLE = replay
        match = read_csv(LOG_SINGLE)
        if match is None:
            continue
        supply = {}
        for side in {"A", "B"}:
            supply[side] = 0
            team = 'team_' + side
            if match[team] is None:
                break
            for unit in match[team]:
                if int(unit) == 85:
                    continue
                unit_val = return_unit_values_by_id(int(unit))
                supply[side] += unit_val['supply']
        if abs(supply["A"]-supply["B"]) < supply_limit and supply["A"] > 0 and supply["B"] > 0:
            close_matchups.append(replay)
            supply_diff.append(abs(supply["A"]-supply["B"]))
            
                #print(replay, supply["A"], supply["B"])
            

            
    print("Close match-ups filtered. %d files qualified." % len(close_matchups))
    return close_matchups, supply_diff
    
def get_remaining_indices(file_count=0, ind1=0, ind2=0, supply=[]):
    file_array = range(0, file_count-1)
    remaining_indices = []
    remaining_supplies = []
    for i in range(len(file_array)):
        if (file_array[i] not in ind1) and file_array[i] not in ind2:
            remaining_indices.append(file_array[i])
            remaining_supplies.append(supply[i])
    return remaining_indices, remaining_supplies

# Function to read one encouter per file .csv-files
def read_csv(csv_file):
    try:
        file = open(csv_file, 'r')
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            if row[0] == '0':
                team_a = row[1]
            elif row[0] == '1':
                team_b = row[1]
            elif row[0] == '2':
                winner = row[1]
                winner = winner.replace('[','').replace(']','')
    except UnicodeDecodeError:
        print(csv_file)
    try:
        dict = {'team_A': string_to_csv(team_a), 'team_B': string_to_csv(team_b), 'winner_code': winner}
    
        file.close()
        return dict
    except:
        # print(csv_file)
        i=1
def string_to_csv(str):
    str = str.replace('[', '').replace(']', '').replace(' ', '')
    arr = str.split(',')
    for elem in arr:
        elem = int(elem.replace("'", ""))
    return arr

def sum_up_csv_files(version = STANDARD_VERSION):
    # get all .csv-files from a given version
    replay_log_files = build_file_array('logs', version)
    match_arr = []
    # create an array which contains all data from given .csv-files
    i = 0
    print('Looking over', len(replay_log_files), 'files')
    while i < len(replay_log_files):
        match_arr.append(read_csv(replay_log_files[i]))
        i = i + 1
    PATH = os.path.join(REPO_DIR, version)
    os.makedirs(PATH, exist_ok=True)
    FILE_PATH = os.path.join(PATH, 'all_csv_from_version_' + version + '.csv')
    # stream all data into a single .csv-file
    with open(FILE_PATH, "w+") as file:
        try:
            for match in match_arr:
                writer = csv.writer(file)
                writer.writerow( (match['team_A'], match['team_B'], match['winner_code']) )
        finally:
            file.close()
# Function for reading the summed up .csv-files (all encounters of one specified version contained in one file)
def read_summed_up_csv(csv_file, maximum = 0):
    match_arr = []
    try:
        file = open(csv_file, 'r')
    except:
        print("Loading failed")
    csv_reader = csv.reader(file, delimiter=',')
    counter = 0 
    for row in csv_reader:
        if counter >= maximum and maximum != 0:
            break;
        try:
            if row[0] == '':
                continue
        except: 
            continue
        i = 0
        first_split = 0
        second_split = 0
        for entry in row:
            if i == 0:
                team_a = entry
            if i == 1:
                team_b = entry
            if i == 2:
                winner = entry
            i += 1
        dict = {'team_A': string_to_csv(team_a), 'team_B': string_to_csv(team_b), 'winner_code': winner}
        match_arr.append(dict)
        counter += 1
       
    file.close()
    return match_arr  

###
def return_unit_values_by_id(id):
    ###terran ids
    if id == 32 or id == 33:
        return siege_tank
    if id == 34 or id == 35:
        return viking
    if id == 45:
        return scv
    if id == 48:
        return marine
    if id == 49:
        return reaper
    if id == 50:
        return ghost
    if id == 51:
        return marauder
    if id == 52:
        return thor
    if id == 53 or id == 484:
        return hellion
    if id == 54:
        return medivac
    if id == 55:
        return banshee
    if id == 56:
        return raven
    if id == 57:
        return battlecruiser
    if id == 268:
        return mule
    if id == 692:
        return cyclone
    ###protoss ids
    if id == 4:
        return colossus
    if id == 10:
        return mothership
    if id == 73:
        return zealot
    if id == 74:
        return stalker
    if id == 75:
        return high_templar
    if id == 76:
        return dark_templar
    if id == 77:
        return sentry
    if id == 78:
        return phoenix
    if id == 79:
        return carrier
    if id == 80:
        return void_ray
    if id == 82:
        return observer
    if id == 83:
        return immortal
    if id == 84:
        return probe
    if id == 141:
        return archon
    if id == 311:
        return adept
    if id == 694:
        return disruptor
    ###zerg ids
    if id == 9:
        return baneling
    if id == 12 or id == 13 or id == 15 or id == 17:
        return changeling
    if id == 104:
        return drone
    if id == 105:
        return zergling
    if id == 106:
        return overlord
    if id == 107:
        return hydralisk
    if id == 108:
        return mutalisk
    if id == 109:
        return ultralisk
    if id == 110:
        return roach
    if id == 111:
        return infestor
    if id == 112:
        return corruptor
    if id == 114:
        return brood_lord
    if id == 126:
        return queen
    if id == 129:
        return overseer
    if id == 289:
        return broodling
    if id == 499:
        return viper 
### helpergfunction for resnet
def get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier
        