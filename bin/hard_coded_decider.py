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





import numpy as np
from absl import app
from data import simulation_pb2

from bin.load_batch import load_batch
from bin.data_visualization import map_id_to_units_race
from bin.util import *

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from lib.unit_constants import *
from lib.config import SCREEN_RESOLUTION, MINIMAP_RESOLUTION, MAP_PATH, \
    REPLAYS_PARSED_DIR, REPLAY_DIR, REPO_DIR
    

def main():
    replay_log_files = []
    match_arr = []
    
    # configure details for execution
    # skip_remis is a boolean to determine whether the loading procedure should ignore samples that resulted in a remis
    # file_version contains the information which version of the samples should be loaded(e.g. the single .csv-File containing all combats, or the .csv-files that contain one encounter each)
    # version declares the version of replay samples to use
    skip_remis = False
    file_version = 'single'
    version = '1_3a'
    
    if file_version == 'multiple':
        replay_log_files = []
        

        replay_log_files = build_file_array('logs', version)
        i = 0
        print('Looking over', len(replay_log_files), 'files')
        while i < len(replay_log_files):
            match_arr.append(read_csv(replay_log_files[i]))
            i = i + 1
            if i%10000 == 0:
                print(i, 'csv-files loaded')

        print('match_arr built...')
    if file_version == 'single':
        file_path = os.path.join(REPO_DIR, 'all_csv_from_version_' + version + '.csv')
        match_arr = read_summed_up_csv(file_path)
    i = 0
    correct_pred = 0
    
    
    
    for match in match_arr:
        if int(match['winner_code']) == 2 and skip_remis:
            continue
        #declare all necessary variables
        type =  {'g', 'a'}
        team = {'A', 'B'}
        powervalue = {}
        hitpoints_and_shield = {}
        unit_types = {}
        attack_type = {}
        invisibility = {}
        detection = {}
        types = {}
        unit_types_ges = {}
        powervalue_summed = {}
        for t in team:
            powervalue[t] = {}
            hitpoints_and_shield[t] = {}
            for ty in type:
                powervalue[t][ty] = 0
                hitpoints_and_shield[t][ty] = 0
            unit_types[t] = {'ground': 0, 'air': 0}
            attack_type[t] = []
            invisibility[t] = False
            detection[t] = False
            # a = armored, p = psionic, m = mechanical, massive, l = light, b = biological, ges sums up all units
            types[t] = {'a' : 0, 'p': 0, 'm': 0, 'massive': 0, 'l': 0, 'b': 0, 'ges': 0}
            unit_types_ges[t] = 0
            powervalue_summed[t] = 0
        
        try:
            #calculate the sum of unit types contained in the army
            for t in team:
                types[t] = calc_attributes(match['team_'+t])
                if t == 'A':
                    ant = 'B'
                else:
                    ant = 'A'
                for ty in type:               
                    #calculate powervalues for each ground and air attacking units considerung the ratio of targetable units in the enemy unit array
                    powervalue[t][ty] = calc_unit_bonus(match['team_'+t], types[ant], ty)            
                # array to get sum of air and ground units
                unit_types[t] = calc_unit_types(match['team_'+t])
                # array to get the sum of contained attack types
                attack_type[t] = calc_attack_types(match['team_'+t])
                # calculate the durability of ground forces as well as air forces
                hitpoints_and_shield[t]['g'], hitpoints_and_shield[t]['a'] = calc_hp_shield_armor(match['team_'+t])

                detection[t], invisibility[t] = get_detection_and_invisibility(match['team_'+t])
                    
        except TypeError:
            print(match, replay_log_files[i])
            continue
        except ZeroDivisionError:
            print(match, replay_log_files[i])
            continue
            
        ### merge powervalues by ratio of enemy unit types
        for t in team:
            unit_types_ges[t] = unit_types[t]['air'] + unit_types[t]['ground']
            if t == 'A':
                ant = 'B'
            else: 
                ant = 'A'
            try: 
                powervalue_summed[t]= (unit_types[ant]['air']/unit_types_ges[ant]) * powervalue[t]['a'] + (unit_types[ant]['ground']/unit_types_ges[ant]) * powervalue[t]['g']
            except ZeroDivisionError:
                print(match, replay_log_files[i])
                continue
        
        ### estimate winner 
        ### Standard value for estimation is REMIS
        estimation = 2   
        
        ### fightable is true for an army if all unit types of the enemy army can be fought by it. 
        ### E.g. If air units are in the opposing army it has to contain units with air targeted attacks, 
        ### If units are invisible the army has to contain units with the detector property a.s.o.
        fightable = {}
        for t in team:
            if t == 'A':
                ant = 'B'
            else: 
                ant = 'A'
            fightable[t] = False
            ### Can the units fight each other
            try:
                fightable[t] = calc_fightable(unit_types[ant], attack_type[t], invisibility[ant], detection[t])
            except TypeError:
                print(match, replay_log_files[i])
                continue
        
        ### estimate outcome comparing powervalues and fighting ability
        estimation = estimateOutcome(fightable['A'], fightable['B'], powervalue_summed['A'], powervalue['A']['g'], powervalue['A']['a'], powervalue_summed['B'], powervalue['B']['g'], powervalue['B']['a'], hitpoints_and_shield['A']['g'], hitpoints_and_shield['A']['a'], hitpoints_and_shield['B']['g'], hitpoints_and_shield['B']['a'])
        i += 1
        if str(estimation) == str(match['winner_code']):
            correct_pred += 1
        print(estimation, match['winner_code'])
    print(correct_pred, i, correct_pred/i)
    
def estimateOutcome(fightable_A, fightable_B, powervalue_A, powervalue_A_ground, powervalue_A_air, powervalue_B, powervalue_B_ground, powervalue_B_air, hitpoints_armor_and_shield_A_ground, hitpoints_armor_and_shield_A_air, hitpoints_armor_and_shield_B_ground, hitpoints_armor_and_shield_B_air):
    ### If units can fight each other the outcome is REMIS    
    if not fightable_A and not fightable_B:
        estimation = 2
    ### If units A can fight units b but not the other way around AND
    ### units A are powerfull enough to clear map within 45s TEAM A WINS 
    ### otherwise REMIS
    if fightable_A and not fightable_B: 
        # times 120 because the map limit per encounter is set to 2 in-game minutes
        if (powervalue_A_ground*60) >= hitpoints_armor_and_shield_B_ground and (powervalue_A_air*60) >= hitpoints_armor_and_shield_B_air:
            estimation = 0
        else: 
            estimation = 2
    ### Same logic applied to Team B 
    if not fightable_A and fightable_B:
        if (powervalue_B_ground*60) >= hitpoints_armor_and_shield_A_ground and (powervalue_B_air*60) >= hitpoints_armor_and_shield_A_air:
            estimation = 1
        else:
            estimation = 2
    ### If both teams can fight each other, we compare the calculated powervalues
    if fightable_A and fightable_B:
        if powervalue_A * (hitpoints_armor_and_shield_A_ground + hitpoints_armor_and_shield_A_air) > powervalue_B * (hitpoints_armor_and_shield_A_ground + hitpoints_armor_and_shield_A_air):
            estimation = 0
        else:
            estimation = 1 
    return estimation

def calc_hp_shield_armor(match):
    hitpoints_and_shield_ground = 0
    hitpoints_and_shield_air = 0
    for id in match:
        id = int(id.replace("'",""))
        if id == 85:
            continue
        unit = return_unit_values_by_id(id)
        if str(unit['type']) == 'g' or str(unit['type']) == 'ga':
            hitpoints_and_shield_ground += unit['hp'] + unit['sh'] + unit['ar']
        if str(unit['type']) == 'a' or str(unit['type']) == 'ga':
            hitpoints_and_shield_air += unit['hp'] + unit['sh'] + unit['ar']
    return hitpoints_and_shield_ground, hitpoints_and_shield_air
    
def get_detection_and_invisibility(match):
    det = False
    inv = False
    for id in match:
        id = int(id.replace("'",""))
        if id == 85:
            continue
        unit = return_unit_values_by_id(id)
        if unit['det'] == 'y':
            det = True
        if unit['inv'] == 'y':
            inv = True
    return det, inv
    
def calc_fightable(unit_types, attack_types, invis, det):
    fightable = False
    if unit_types['ground'] > 0:
        if 'g' in attack_types or 'ga' in attack_types:
            fightable = True 
        else: 
            fightable = False
    if unit_types['air'] > 0:
        if 'a' in attack_types or 'ga' in attack_types:
            fightable= True
        else: 
            fightable = False
    if invis:
        if det:
            fightable = fightable
        else: fightable = False
    return fightable


def calc_attack_types(match):
    attack_type = []
    for id in match:
        id = int(id.replace("'",""))
        if id == 85:
            continue
        unit = return_unit_values_by_id(id)
        
        if unit['target'] in attack_type:
            ###
            attack_type = attack_type
        else:
            attack_type.append(unit['target'])
    return attack_type
        
   
def calc_unit_types(match):
    unit_type = {'ground': 0, 'air': 0}
    for id in match:
        id = int(id.replace("'",""))
        if id == 85:
            continue
        unit = return_unit_values_by_id(id)
        if unit['type'] == 'a':
        ###
            unit_type['air'] += 1
        elif unit['type'] == 'g':
            unit_type['ground'] += 1
        elif unit['type'] == 'ga':
            unit_type['ground'] += 1
            unit_type['air'] += 1
    return unit_type
 
def calc_unit_bonus(match, t_B, ga):
    #construct pw_% value
    unit_power = 'pw_' + ga
    powervalue = 0
    for id in match:
        id = int(id.replace("'",""))
        if id == 85:
            continue
        unit = return_unit_values_by_id(id)
        bonus = 0
        if unit[unit_power] == 0:
            unit[unit_power] = 1
        if unit['sh'] == 0:
            unit['sh'] = 1  
        try:
            if unit['bonus'][2] == ga or unit['bonus'][2] == 'ga':
                if unit['bonus'][0] == 'a':
                    bonus += (unit['bonus'][1] * (t_B['a']/t_B['ges']))
                if unit['bonus'][0] == 'p':
                    bonus += (unit['bonus'][1] * (t_B['p']/t_B['ges']))
                if unit['bonus'][0] == 'm':
                    bonus += (unit['bonus'][1] * (t_B['m']/t_B['ges']))
                if unit['bonus'][0] == 'massive':
                    bonus += (unit['bonus'][1] * (t_B['massive']/t_B['ges']))
                if unit['bonus'][0] == 'l':
                    bonus += (unit['bonus'][1] * (t_B['l']/t_B['ges']))
                if unit['bonus'][0] == 'b':
                    bonus += (unit['bonus'][1] * (t_B['b']/t_B['ges']))
        except:
            bonus = 0
        powervalue += (unit[unit_power] + bonus)
        
    return powervalue

def calc_attributes(match):
    t_A = {'a' : 0, 'p': 0, 'm': 0, 'massive': 0, 'l': 0, 'b': 0, 'ges': 0}
    for id in match:
        id = int(id.replace("'",""))
        if id == 85:
            continue
        unit = return_unit_values_by_id(id)
        for t in unit['attributes']:
            if t == 'a':
                t_A['a'] += 1
            if t == 'p':
                t_A['p'] += 1
            if t == 'm':
                t_A['m'] += 1
            if t == 'massive':
                t_A['massive'] += 1
            if t == 'l':
                t_A['l'] += 1
            if t == 'b':
                t_A['b'] += 1
        t_A['ges'] += 1
    return t_A
    
if __name__ == "__main__":
    main()

