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

from bin.util import read_csv

import pandas as pd
import numpy as np
from absl import app
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from lib.config import SCREEN_RESOLUTION, MINIMAP_RESOLUTION, MAP_PATH, \
    REPLAYS_PARSED_DIR, REPLAY_DIR, REPO_DIR
    
terran_ids = [49,#reaper
              51,#Marauder
              48,#Marine
              50,#Ghost
              268,#Mule
              45,#SCV
              56,#Raven
              55,#Banshee
              54,#Medivac
              34,35,#Viking
              57,#Battlecruiser
              52,#Thor
              32,33,#Siege Tank
              53,484,#Hellion
              692#Cyclone
              ]
protoss_ids = [10,#Mutterschiff 
               80,#Void Ray 
               76,#Dark Templar 
               79,#Carrier 
               4,#Colossus 
               73,#Zealot 
               84,#Probe 
               78,#Phoenix 
               141,#Archon 
               83,#Immortal 
               82,#Observer 
               75,#High Templar
               74,#Stalker
               694,#Disruptor
               77,#Sentry
               85,#Interceptor
              311#Adept
              ]

zerg_ids = [111,#Verseucher
            112,#Schänder
            129,#Overseer
            105,#Zergling 
            114,#Brutlord
            107,#Hydralisk
            126,#Königin
            108,#Mutalisk
            109,#Ultralisk
            110,#Roach//Schabe
            104,#Drohne
            106,#Overlord
            13,#Formling_Zealot
            17,#Formling_Zergling
            15,#Formling_Marine
            9,#Berstling
            289,#Brütling
            499,#Viper
            12#Formling
            ]
    
    
def main(unsued_args):
    replay_log_files = []
    match_arr = []
    count_race_wins = {'zerg_count': 0,
                       'zerg_win': 0,
                       'zerg_win_prot': 0,
                       'zerg_win_terr': 0,
                       'zerg_win_zerg': 0,
                       'prot_count': 0,
                       'prot_win': 0,
                       'prot_win_zerg': 0,
                       'prot_win_terr': 0,
                       'prot_win_prot': 0,
                       'terr_count': 0,
                       'terr_win': 0,
                       'terr_win_prot': 0,
                       'terr_win_zerg': 0,
                       'terr_win_terr': 0,
                       'avg_unit_count': 0
                       }
    for root, dir, files in os.walk(os.path.join(REPO_DIR, 'log')):
        for file in files:
            if file.endswith(".csv"):
                replay_log_files.append(os.path.join(root, file))
    i = 0
    print('Looking over', len(replay_log_files), 'files')
    while i < len(replay_log_files):
        match_arr.append(read_csv(replay_log_files[i]))
        i = i + 1
        if i%10000 == 0:
            print(i, 'csv-files loaded')
    i = 0
    print('match_arr built...', match_arr[0])
    print(len(terran_ids))
    while i < len(match_arr):
        #Get the races by Names
        race_A = map_id_to_units_race(match_arr[i]['team_A'][0])
        race_B = map_id_to_units_race(match_arr[i]['team_B'][0])
        # if race_A == 'No Match':
            # print('Team A', replay_log_files[i], str(match_arr[i]['team_A']))
        # if race_B == 'No_Match':
            # print('Team B', replay_log_files[i], str(match_arr[i]['team_B']))
        #Get the winning faction
        if match_arr[i]['winner_code'] == '0':
            winner_race = race_A
        elif match_arr[i]['winner_code'] == '1':
            winner_race = race_B
        else: winner_race = 'Remis'
        
        #print(match_arr[i])
        #Compare the statistics for races and collect them
        #Results for Zerg
        if race_A == 'Zerg' or race_B == 'Zerg':
            count_race_wins['zerg_count'] += 1
        if race_A == 'Zerg' and match_arr[i]['winner_code'] == '0':
            count_race_wins['zerg_win'] += 1
            if race_B == 'Protoss':
                count_race_wins['zerg_win_prot'] += 1
            if race_B == 'Terran':
                count_race_wins['zerg_win_terr'] += 1
            if race_B == 'Zerg':
                count_race_wins['zerg_win_zerg'] += 1
        if race_B == 'Zerg' and match_arr[i]['winner_code'] == '1':
            count_race_wins['zerg_win'] += 1
            if race_A == 'Protoss':
                count_race_wins['zerg_win_prot'] += 1
            if race_A == 'Terran':
                count_race_wins['zerg_win_terr'] += 1
            if race_A == 'Zerg':
                count_race_wins['zerg_win_zerg'] += 1
                
        #Results for Protoss
        if race_A == 'Protoss' or race_B == 'Protoss':
            count_race_wins['prot_count'] += 1
        if race_A == 'Protoss' and match_arr[i]['winner_code'] == '0':
            count_race_wins['prot_win'] += 1
            if race_B == 'Zerg':
                count_race_wins['prot_win_zerg'] += 1
            if race_B == 'Terran':
                count_race_wins['prot_win_terr'] += 1
            if race_B == 'Protoss':
                count_race_wins['prot_win_prot'] += 1
        if race_B == 'Protoss' and match_arr[i]['winner_code'] == '1':
            count_race_wins['prot_win'] += 1
            if race_A == 'Zerg':
                count_race_wins['prot_win_zerg'] += 1
            if race_A == 'Terran':
                count_race_wins['prot_win_terr'] += 1
            if race_A == 'Protoss':
                count_race_wins['prot_win_prot'] += 1
                
        #Results for Terran
        if race_A == 'Terran' or race_B == 'Terran':
            count_race_wins['terr_count'] += 1
        if race_A == 'Terran' and match_arr[i]['winner_code'] == '0':
            count_race_wins['terr_win'] += 1
            if race_B == 'Zerg':
                count_race_wins['terr_win_zerg'] += 1
            if race_B == 'Protoss':
                count_race_wins['terr_win_prot'] += 1
            if race_B == 'Terran':
                count_race_wins['terr_win_terr'] += 1
        if race_B == 'Terran' and match_arr[i]['winner_code'] == '1':
            count_race_wins['terr_win'] += 1
            if race_A == 'Zerg':
                count_race_wins['terr_win_zerg'] += 1
            if race_A == 'Protoss':
                count_race_wins['terr_win_prot'] += 1    
            if race_A == 'Terran':
                count_race_wins['terr_win_terr'] += 1
        i = i + 1
     
    print(count_race_wins)
    n_groups = 3
    
    val_count = (count_race_wins['prot_count'], count_race_wins['terr_count'], count_race_wins['zerg_count'])
    val_win = (count_race_wins['prot_win'], count_race_wins['terr_win'], count_race_wins['zerg_win'])
    val_win_ag_zerg = (count_race_wins['prot_win_zerg'], count_race_wins['terr_win_zerg'], count_race_wins['zerg_win_zerg'])
    val_win_ag_prot = (count_race_wins['prot_win_prot'], count_race_wins['terr_win_prot'], count_race_wins['zerg_win_prot'])
    val_win_ag_terr = (count_race_wins['prot_win_terr'], count_race_wins['terr_win_terr'], count_race_wins['zerg_win_terr'])
    
    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.1

    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    
    rects1 = ax.bar(index, val_count, bar_width, alpha=opacity, color='b', error_kw=error_config, label='Matchcount[Protoss|Terran|Zerg]')
    rects2 = ax.bar(index + bar_width, val_win, bar_width, alpha=opacity, color='r', error_kw=error_config, label='Wincount[P|T|Z]')
    rects3 = ax.bar(index + 2*bar_width, val_win_ag_zerg, bar_width, alpha=opacity, color='y', error_kw=error_config, label='Wincount Against Zerg[P|T|Z]')
    rects4 = ax.bar(index + 3*bar_width, val_win_ag_prot, bar_width, alpha=opacity, color='g', error_kw=error_config, label='Wincount Against Protoss[P|T|Z]')
    rects5 = ax.bar(index + 4*bar_width, val_win_ag_terr, bar_width, alpha=opacity, color='c', error_kw=error_config, label='Wincount Against Terran[P|T|Z]')
    
    ax.set_xlabel('Races')
    ax.set_ylabel('Counts')
    ax.set_title('Count of encounters and wins per race')
    ax.set_xticks(index + bar_width/5)
    ax.set_xticklabels(('Protoss', 'Terran', 'Zerg'))
    ax.legend()
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)


    fig.tight_layout()
    plt.show()
    
def map_id_to_units_race(single_id):
    if isInArray(single_id, terran_ids):
        return 'Terran'
    elif isInArray(single_id, protoss_ids):
        return 'Protoss'
    elif isInArray(single_id, zerg_ids):
        return 'Zerg'
    else: 
        return 'No Match'
  
def isInArray(idsingle, arr):
    i = 0
    while i < len(arr):
        if str(arr[i]) == idsingle:
            return True
        i = i + 1
    return False
    

def map_ids_to_units(ids, race):
    name_arr = []
    if race == 'Terran':
        for id in ids:
            if id == 35:
                name_arr.append('Viking')
            else: name_arr.append(str(id))
                
    if race == 'Protoss':
        for id in ids:
            if id == 10:
                name_arr.append('Mothership')
            elif id == 80:
                name_arr.append('Void Ray')
            elif id == 76:
                name_arr.append('Dark Templar')
            elif id == 79:
                name_arr.append('Carrier')
            elif id == 4:
                name_arr.append('Colossus')
            elif id == 73:
                name_arr.append('Zealot')
            elif id == 84:
                name_arr.append('Probe')
            elif id == 78:
                name_arr.append('Phoenix')
            elif id == 141:
                name_arr.append('Archon')
            else: name_arr.append(str(id))
            
    if race == 'Zerg':
        for id in ids:
            if id == 111:
                name_arr.append('Infestor')
            elif id == 129:
                name_arr.append('Overseer')
            else: name_arr.append(str(id))    
    return name_arr
def entry_point():
    app.run(main)


if __name__ == '__main__':
    app.run(main)