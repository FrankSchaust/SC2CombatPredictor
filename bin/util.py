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
import pandas as pd
from absl import app

from pysc2 import run_configs
from pysc2.bin.play import get_replay_version
from s2clientprotocol.common_pb2 import Size2DI
from s2clientprotocol.sc2api_pb2 import InterfaceOptions, RequestStartReplay, \
    SpatialCameraSetup

from bin.read_csv import *    
from lib.config import SCREEN_RESOLUTION, MINIMAP_RESOLUTION, MAP_PATH, \
    REPLAYS_PARSED_DIR, REPLAY_DIR, REPO_DIR, STANDARD_VERSION
from data.simulation_pb2 import Battle, Simulation

# Input arguments: 
# type = String with associated files to load
#       Either: 'replays' to build a file_array containing the parsed_replays names
#       Or: 'logs' to build a file_array containing the logs names
# version = String to define the version of the data; refers to the suffix of the folders specified 'log_v1_3' is specified by '_v1_3'
# Return: Array with all files in the given directory 
def build_file_array(type = 'replays', version = STANDARD_VERSION):
    file_array = []
    # Routine for replay data
    if type == 'replays':
        if version == '':
            DIRECTORY = os.path.join(REPLAYS_PARSED_DIR)
        else:
            DIRECTORY = os.path.join(REPLAYS_PARSED_DIR, version)
        print("Creating list of used files")
        for root, dir, files in os.walk(DIRECTORY):
            for file in files:
                if file.endswith(".SC2Replay_parsed.gz"):
                    file_array.append(os.path.join(root, file))
        print("Available Files: ", len(file_array))
    # Routine for log data    
    if type == 'logs':
        if version == '':
            PATH = os.path.join(REPO_DIR, 'log')
        else:
            PATH = os.path.join(REPO_DIR, 'log', version)
        for root, dir, files in os.walk(PATH):
            for file in files:
                if file.endswith(".csv"):
                    file_array.append(os.path.join(root, file))
    return file_array
    
def play_replay(dir, version):
    simulation = Simulation()
    PATH = os.path.join(REPO_DIR, 'CombatGenerator-v' + version + '.SC2Map')
    print('Going to parse replay "{}".'.format(dir),
                  file=sys.stderr)
                  
    run_config = run_configs.get()
    replay_data = run_config.replay_data(dir)
    with run_config.start(game_version=get_replay_version(replay_data),
                                      full_screen=False) as controller:
                    print('Starting replay...', file=sys.stderr)
                    controller.start_replay(RequestStartReplay(
                        replay_data=replay_data,
                        # Overwrite map_data because we know where our map is located
                        # (so we don't care about the map path stored in the replay).
                        map_data=run_config.map_data(PATH),
                        # Observe from Team Minerals so player_relative features work.
                        observed_player_id=1,
                        # Controls what type of observations we will receive
                        options=InterfaceOptions(
                            # Raw observations include absolute unit owner, position,
                            # and type we enable them for constructing a baseline.
                            raw=True,
                            # Score observations include statistics collected over the
                            # game, most aren't applicable to this map, so disable them.
                            score=False,
                            # Feature layer observations are those described in the
                            # SC2LE paper. We want to learn on those, so enable them.
                            feature_layer=SpatialCameraSetup(
                                width=24,
                                resolution=Size2DI(x=SCREEN_RESOLUTION,
                                                   y=SCREEN_RESOLUTION),
                                minimap_resolution=Size2DI(x=MINIMAP_RESOLUTION,
                                                           y=MINIMAP_RESOLUTION))),
                        # Ensure that fog of war won't hinder observations.
                        disable_fog=True))
                    error_flag = False
                    obs = controller.observe()
                    while not obs.player_result:
                        # We advance 16 steps at a time here because this is the largest
                        # value which still ensures that we will have at least one
                        # observation between different phases (units spawn, units
                        # engage, outcome determined, units remove) when wait time is
                        # set to 1.0 game seconds in editor.
                        controller.step(8)
                        time.sleep(1)
                        obs = controller.observe()
# Function to read one encouter per file .csv-files
def read_csv(csv_file):

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
    dict = {'team_A': string_to_csv(team_a), 'team_B': string_to_csv(team_b), 'winner_code': winner}
    file.close()
    return dict

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