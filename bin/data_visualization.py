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

from bin.util import read_csv, build_file_array, filter_close_matchups, return_unit_values_by_id

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
    match_arr = []
    count_race_wins = {'zerg_count': 0,
                       'zerg_win': 0,
                       'zerg_win_protoss': 0,
                       'zerg_win_terran': 0,
                       'zerg_win_zerg': 0,
                       'protoss_count': 0,
                       'protoss_win': 0,
                       'protoss_win_zerg': 0,
                       'protoss_win_terran': 0,
                       'protoss_win_protoss': 0,
                       'terran_count': 0,
                       'terran_win': 0,
                       'terran_win_protoss': 0,
                       'terran_win_zerg': 0,
                       'terran_win_terran': 0,
                       'avg_unit_count': 0
                       }
    match_arr = []
    logs = []
    logs = build_file_array(type='logs', version=['1_3d', '1_3d_10sup', '1_3d_15sup'])
    for l in logs:
        match = read_csv(l)
        if match is None: 
            continue 
        if ((match['team_A'] is None) and (match['team_B'] is None)):
            continue
        match_arr.append(match)
    print(len(match_arr))
    supplies = {'A': [], 'B': []}
    CloseWins = {'zerg': 0, 'protoss': 0, 'terran': 0}
    HighDiffWins = {'zerg': 0, 'protoss': 0, 'terran': 0}
    EvenWins = {'zerg': 0, 'protoss': 0, 'terran': 0}
    BehindWins = {'zerg': 0, 'protoss': 0, 'terran': 0}
    Remis = {'high A': 0, 'high B': 0, 'even': 0, 'close A': 0, 'close B': 0}
    count = 0
    for m in match_arr:
        #Get the races by Names
        race_A = map_id_to_units_race(m['team_A'][0])
        race_B = map_id_to_units_race(m['team_B'][0])

        supply = {'A': 0, 'B': 0}
        #Get supplies for each side 
        for side in {"A", "B"}:
            supply[side] = 0
            team = 'team_' + side
            if m[team] is None:
                break
            for unit in m[team]:
                if int(unit) == 85:
                    continue
                unit_val = return_unit_values_by_id(int(unit))
                supply[side] += unit_val['supply']
            supplies[side].append(supply[side])
        
        winCond = detWinsCond(supply)
        if (supply['A'] - supply['B'] > 10 and str(m['winner_code']) == 1) or (supply['A'] - supply['B'] < -10 and str(m['winner_code']) == 0):
            count += 1
        try:
            wc = str(m['winner_code'])
            if wc == '2':
                Remis[winCond] += 1
            else:
                if wc == '2':
                    continue
                if winCond == 'even':
                    if wc == '0':
                        EvenWins[race_A.lower()] += 1
                    else:
                        EvenWins[race_B.lower()] += 1
                if winCond == 'high A':
                    if wc == '0':
                        HighDiffWins[race_A.lower()] += 1
                    else:
                        BehindWins[race_B.lower()] += 1
                if winCond == 'close A':
                    if wc == '0':
                        CloseWins[race_A.lower()] += 1
                    else:
                        BehindWins[race_B.lower()] += 1
                if winCond == 'high B':
                    if wc == '1':
                        HighDiffWins[race_B.lower()] += 1
                    else:
                        BehindWins[race_A.lower()] += 1
                if winCond == 'close B':
                    if wc == '1':
                        CloseWins[race_B.lower()] += 1
                    else:
                        BehindWins[race_A.lower()] += 1
            # print(wc, winCond, BehindWins)
        except KeyError:
            print(supply, winCond)
        # if race_A == 'No Match':
            # print('Team A', replay_log_files[i], str(match_arr[i]['team_A']))
        # if race_B == 'No_Match':
            # print('Team B', replay_log_files[i], str(match_arr[i]['team_B']))
        #Get the winning faction

        #Compare the statistics for races and collect them
        #Results for Zerg

        
        if race_A == race_B:
            count_race_wins[race_A.lower()+'_count'] += 1
        else:
            count_race_wins[race_A.lower()+'_count'] += 1
            count_race_wins[race_B.lower()+'_count'] += 1

        if wc == '0':
            count_race_wins[race_A.lower()+'_win'] += 1
            count_race_wins[race_A.lower()+'_win_'+race_B.lower()] += 1
        if wc == '1':
            count_race_wins[race_B.lower()+'_win'] += 1
            count_race_wins[race_B.lower()+'_win_'+race_A.lower()] += 1
        
    diffs = []
    diffCount = {}
    print(count)
    # print(len(supplies["A"]), len(supplies["B"]))
    for s in range(len(supplies["A"])):
        diffs.append(abs(supplies["A"][s]-supplies["B"][s]))
    uniques = np.unique(diffs)
    for u in uniques:
        diffCount[str(u*2)] = diffs.count(u)
    #print(uniques)
    #print(diffCount)
    #print(count_race_wins)
    #print(CloseWins, HighDiffWins, BehindWins, EvenWins, Remis)
    n_groups = 3
    
    val_count = (count_race_wins['protoss_count'], count_race_wins['terran_count'], count_race_wins['zerg_count'])
    val_win = (count_race_wins['protoss_win'], count_race_wins['terran_win'], count_race_wins['zerg_win'])
    val_win_ag_zerg = (count_race_wins['protoss_win_zerg'], count_race_wins['terran_win_zerg'], count_race_wins['zerg_win_zerg'])
    val_win_ag_prot = (count_race_wins['protoss_win_protoss'], count_race_wins['terran_win_protoss'], count_race_wins['zerg_win_protoss'])
    val_win_ag_terr = (count_race_wins['protoss_win_terran'], count_race_wins['terran_win_terran'], count_race_wins['zerg_win_terran'])
    
    zerg = (CloseWins['zerg'], BehindWins['zerg'], HighDiffWins['zerg'], EvenWins['zerg'])
    protoss = (CloseWins['protoss'], BehindWins['protoss'], HighDiffWins['protoss'], EvenWins['protoss'])
    terran = (CloseWins['terran'], BehindWins['terran'], HighDiffWins['terran'], EvenWins['terran'])
    # print(zerg)
    fig = plt.figure()
    

    # ax = plt.subplot()
    # bx = plt.subplot()
    cx = plt.subplot()

    vals = []
    for m in uniques:
        vals.append(diffCount[str(m*2)])
    index = np.arange(n_groups)
    index_2 = np.arange(len(uniques))
    index_3 = np.arange(4)
    bar_width = 0.1

    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    
    # rects1 = ax.bar(index, val_count, bar_width, alpha=opacity, color='b', error_kw=error_config, label='Anzahl Gefechte [Protoss|Terran|Zerg]')
    # rects2 = ax.bar(index + bar_width, val_win, bar_width, alpha=opacity, color='r', error_kw=error_config, label='Anzahl Siege [P|T|Z]')
    # rects3 = ax.bar(index + 2*bar_width, val_win_ag_zerg, bar_width, alpha=opacity, color='y', error_kw=error_config, label='Anzahl Siege gegen Zerg [P|T|Z]')
    # rects4 = ax.bar(index + 3*bar_width, val_win_ag_prot, bar_width, alpha=opacity, color='g', error_kw=error_config, label='Anzahl Siege gegen Protoss [P|T|Z]')
    # rects5 = ax.bar(index + 4*bar_width, val_win_ag_terr, bar_width, alpha=opacity, color='c', error_kw=error_config, label='Anzahl Siege gegen Terran [P|T|Z]')
    
    # ax.set_xlabel('Rasse')
    # ax.set_ylabel('Anzahl')
    # ax.set_title('Anzahl der Gefechte und Siege per Rasse')
    # ax.set_xticks(index + bar_width/5)
    # ax.set_xticklabels(('Protoss', 'Terran', 'Zerg'))
    # ax.legend()
    
    
    # bx.bar(index_2, vals, alpha=opacity, error_kw=error_config)
    # bx.set_xlabel('Supply-Differenzen')
    # bx.set_ylabel('Anzahl Gefechte')


    c_rects1 = cx.bar(index_3, zerg, bar_width, alpha=opacity, color='b', error_kw=error_config, label='Zerg')
    c_rects2 = cx.bar(index_3 + bar_width, protoss, bar_width, alpha=opacity, color='r', error_kw=error_config, label='Protoss')
    c_rects3 = cx.bar(index_3 + bar_width*2, terran, bar_width, alpha=opacity, color='y', error_kw=error_config, label='Terran')

    cx.set_ylabel('Anzahl Siege')
    cx.set_xticks(index_3 + bar_width/4)
    cx.set_xticklabels(('Sieg bei kleiner Differenz', 'Sieg trotz Nachteil', 'Sieg mit hoher Differenz', 'Sieg bei gleichem Supply'))
    cx.legend()

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    # autolabel(rects1, ax)
    # autolabel(rects2, ax)
    # autolabel(rects3, ax)
    # autolabel(rects4, ax)
#     autolabel(rects5, ax)

    autolabel(c_rects1, cx)
    autolabel(c_rects2, cx)
    autolabel(c_rects3, cx)

    fig.tight_layout()
    plt.show()

def detWinsCond(supply):
    diff = supply['A'] - supply['B']
    if diff == 0:
        return 'even'
    #Close Matchups
    if diff <= 5 and diff > 0:
        return 'close A'
    if diff < 0 and diff >= -5:
        return 'close B'
    if diff > 5:
        return 'high A'
    if diff < -5:
        return 'high B'

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