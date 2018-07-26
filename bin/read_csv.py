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
import csv

import pandas as pd
import numpy as np
from absl import app

from lib.config import SCREEN_RESOLUTION, MINIMAP_RESOLUTION, MAP_PATH, \
    REPLAYS_PARSED_DIR, REPLAY_DIR, REPO_DIR
    

def main(unused_args):

    print(read_csv(os.path.join(REPO_DIR, 'log', 'SC2CombatGenerator01-05-2018_00-11-46', 'unit_log_SC2CombatGenerator01-05-2018_00-11-46_3.csv')))

    
    
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
    return arr
	

def entry_point():
    app.run(main)


if __name__ == '__main__':
    app.run(main)