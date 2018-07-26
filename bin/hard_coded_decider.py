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

from bin.read_csv import read_csv
from bin.load_batch import load_batch
from bin.data_visualization import map_id_to_units_race

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from lib.unit_constants import *
from lib.config import SCREEN_RESOLUTION, MINIMAP_RESOLUTION, MAP_PATH, \
    REPLAYS_PARSED_DIR, REPLAY_DIR, REPO_DIR
    

def main():
    replay_log_files = []
    match_arr = []

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
    correct_pred = 0
    print('match_arr built...')
    
    
    
    for match in match_arr:
        # if match['winner_code'] == '2':
            # continue
        #match = match_arr[179]
        #print(match)
        powervalue_A_g = 0
        powervalue_A_a = 0
        powervalue_B_g = 0
        powervalue_B_a = 0
        ut_A = {'ground': 0, 'air': 0}
        ut_B = {'ground': 0, 'air': 0}
        at_A = []
        at_B = []
        
        invis_A = False
        invis_B = False
        det_A = False
        det_B = False
        t_A = {'a' : 0, 'p': 0, 'm': 0, 'massive': 0, 'l': 0, 'b': 0, 'ges': 0}
        t_B = {'a' : 0, 'p': 0, 'm': 0, 'massive': 0, 'l': 0, 'b': 0, 'ges': 0}
        try:
            for id in match['team_A']:
                if str(id) == '85':
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
            
            for id in match['team_B']:
                if str(id) == '85':
                    continue
                unit = return_unit_values_by_id(id)
                for t in unit['attributes']:
                    if t == 'a':
                        t_B['a'] += 1
                    if t == 'p':
                        t_B['p'] += 1
                    if t == 'm':
                        t_B['m'] += 1
                    if t == 'massive':
                        t_B['massive'] += 1
                    if t == 'l':
                        t_B['l'] += 1
                    if t == 'b':
                        t_B['b'] += 1
                t_B['ges'] += 1
                    
                    
            for id in match['team_A']:
                if str(id) == '85':
                    continue
                unit = return_unit_values_by_id(id)
                bonus = 0
                if unit['pw_g'] == 0:
                    unit['pw_g'] = 1
                if unit['sh'] == 0:
                    unit['sh'] = 1  
                try:
                    if unit['bonus'][2] == 'g' or unit['bonus'][2] == 'ga':
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
                    
                powervalue_A_g += (unit['pw_g'] + bonus) * (1+unit['hp']/10) * (1 + unit['ar']/10) * (1+unit['sh']/10)
                
                bonus = 0
                if unit['pw_a'] == 0:
                    unit['pw_a'] = 1
                if unit['sh'] == 0:
                    unit['sh'] = 1
                try:
                    if unit['bonus'][2] == 'a' or unit['bonus'][2] == 'ga':
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
                    
                powervalue_A_a += unit['pw_a'] * (1+unit['hp']/10) * (1 + unit['ar']/10) * (1+unit['sh']/10)
                if unit['type'] == 'a':
                    ###
                    ut_A['air'] += 1
                elif unit['type'] == 'g':
                    ut_A['ground'] += 1
                elif unit['type'] == 'ga':
                    ut_A['ground'] += 1
                    ut_A['air'] += 1
                 
                if unit['target'] in at_A:
                    ###
                    at_A = at_A
                else:
                    at_A.append(unit['target'])

                if unit['det'] == 'y':
                    det_A = True
                
                if unit['inv'] == 'y':
                    invis_A = True
            #print(powervalue_A_g, powervalue_A_a, ut_A, at_A, det_A, invis_A)
            for id in match['team_B']:
                if str(id) == '85':
                    continue
                unit = return_unit_values_by_id(id)
                bonus = 0
                if unit['pw_g'] == 0:
                    unit['pw_g'] = 1
                if unit['sh'] == 0:
                    unit['sh'] = 1
                try:
                    if unit['bonus'][2] == 'g' or unit['bonus'][2] == 'ga':
                        if unit['bonus'][0] == 'a':
                            bonus += (unit['bonus'][1] * (t_A['a']/t_A['ges']))
                        if unit['bonus'][0] == 'p':
                            bonus += (unit['bonus'][1] * (t_A['p']/t_A['ges']))
                        if unit['bonus'][0] == 'm':
                            bonus += (unit['bonus'][1] * (t_A['m']/t_A['ges']))
                        if unit['bonus'][0] == 'massive':
                            bonus += (unit['bonus'][1] * (t_A['massive']/t_A['ges']))
                        if unit['bonus'][0] == 'l':
                            bonus += (unit['bonus'][1] * (t_A['l']/t_A['ges']))
                        if unit['bonus'][0] == 'b':
                            bonus += (unit['bonus'][1] * (t_A['b']/t_A['ges']))
                except:
                    bonus = 0
                    
                powervalue_B_g += unit['pw_g'] * (1+unit['hp']/10) * (1 + unit['ar']/10) * (1+unit['sh']/10)
                
                bonus = 0
                if unit['pw_a'] == 0:
                    unit['pw_a'] = 1
                if unit['sh'] == 0:
                    unit['sh'] = 1
                try:
                    if unit['bonus'][2] == 'a' or unit['bonus'][2] == 'ga':
                        if unit['bonus'][0] == 'a':
                            bonus += (unit['bonus'][1] * (t_A['a']/t_A['ges']))
                        if unit['bonus'][0] == 'p':
                            bonus += (unit['bonus'][1] * (t_A['p']/t_A['ges']))
                        if unit['bonus'][0] == 'm':
                            bonus += (unit['bonus'][1] * (t_A['m']/t_A['ges']))
                        if unit['bonus'][0] == 'massive':
                            bonus += (unit['bonus'][1] * (t_A['massive']/t_A['ges']))
                        if unit['bonus'][0] == 'l':
                            bonus += (unit['bonus'][1] * (t_A['l']/t_A['ges']))
                        if unit['bonus'][0] == 'b':
                            bonus += (unit['bonus'][1] * (t_A['b']/t_A['ges']))
                except:
                    bonus = 0
                    
                powervalue_B_a += unit['pw_a'] * (1+unit['hp']/10) * (1 + unit['ar']/10) * (1+unit['sh']/10)
                
                if unit['type']== 'a':
                    ###
                    ut_B['air'] += 1
                elif unit['type'] == 'g':
                    ut_B['ground'] += 1
                elif unit['type'] == 'ga':
                    ut_B['ground'] += 1
                    ut_B['air'] += 1
                if unit['target'] in at_B:
                    ###
                    at_B = at_B
                else:
                    at_B.append(unit['target'])

                
                if unit['det'] == 'y':
                    det_B = True
                
                if unit['inv'] == 'y':
                    invis_B = True
                    
            #print(powervalue_B_g, powervalue_B_a, ut_B, at_B, det_B, invis_B)
        except TypeError:
            print(match, replay_log_files[i])
            continue
        except ZeroDivisionError:
            print(match, replay_log_files[i])
            continue
            
        ### merge powervalues by ratio of enemy unit types
        
        ut_A_ges = ut_A['air'] + ut_A['ground']
        ut_B_ges = ut_B['air'] + ut_B['ground']
        try: 
            powervalue_A = (ut_B['air']/ut_B_ges) * powervalue_A_a + (ut_B['ground']/ut_B_ges) * powervalue_A_g
            powervalue_B = (ut_A['air']/ut_A_ges) * powervalue_B_a + (ut_A['ground']/ut_A_ges) * powervalue_B_g 
        except ZeroDivisionError:
            print(match, replay_log_files[i])
            continue
        ### Standard value for estimation is REMIS
        
        
        estimation = 2            
        ###estimate winner     
        fightable_A = False
        fightable_B = False
        ### Can the units fight each other
        if ut_A['ground'] > 0:
            if 'g' in at_B or 'ga' in at_B:
                fightable_B = True 
            else: 
                fightable_B = False
        if ut_A['air'] > 0:
            if 'a' in at_B or 'ga' in at_B:
                fightable_B = True
            else: 
                fightable_B = False
        
        if ut_B['ground'] > 0: 
            if 'g' in at_A or 'ga' in at_A:
                fightable_A = True  
            else: 
                fightable_A = False
        if ut_B['air'] > 0: 
            if 'a' in at_A or 'ga' in at_A:
                fightable_A = True
            else: 
                fightable_A = False
        
        ### Check for invisibility
        if invis_A:
            if det_B: 
                fightable_B = fightable_B
            else: fightable_B = False
        if invis_B:
            if det_A:
                fightable_A = fightable_A
            else: fightable_A = False
            
        ### If units can fight each other the outcome is REMIS    
        if not fightable_A and not fightable_B:
            estimation = 2
        ### If units A can fight units b but not the other way around AND
        ### units A are powerfull enough to clear map within 45s TEAM A WINS 
        ### otherwise REMIS
        if fightable_A and not fightable_B: 
            if powervalue_A/2 > powervalue_B:
                estimation = 0
            else: 
                estimation = 2
        ### Same logic applied to Team B 
        if not fightable_A and fightable_B:
            if powervalue_A < powervalue_B/2:
                estimation = 1
            else:
                estimation = 2
        ### If both teams can fight each other, we compare the calculated powervalues
        if fightable_A and fightable_B:
            if powervalue_A > powervalue_B:
                estimation = 0
            else:
                estimation = 1 
        
       
        print(powervalue_A, fightable_A) 
        print(powervalue_B, fightable_B) 
        print(estimation, match['winner_code'], i)
        print()
        i += 1
        if str(estimation) == str(match['winner_code']):
            correct_pred += 1
    
    print(correct_pred, i, correct_pred/i)
    
def return_unit_values_by_id(id):
    ###terran ids
    if str(id) == '32' or str(id) == '33':
        return siege_tank
    if str(id) == '34' or str(id) == '35':
        return viking
    if str(id) == '45':
        return scv
    if str(id) == '48':
        return marine
    if str(id) == '49':
        return reaper
    if str(id) == '50':
        return ghost
    if str(id) == '51':
        return marauder
    if str(id) == '52':
        return thor
    if str(id) == '53' or str(id) == '484':
        return hellion
    if str(id) == '54':
        return medivac
    if str(id) == '55':
        return banshee
    if str(id) == '56':
        return raven
    if str(id) == '57':
        return battlecruiser
    if str(id) == '268':
        return mule
    if str(id) == '692':
        return cyclone
    ###protoss ids
    if str(id) == '4':
        return colossus
    if str(id) == '10':
        return mothership
    if str(id) == '73':
        return zealot
    if str(id) == '74':
        return stalker
    if str(id) == '75':
        return high_templar
    if str(id) == '76':
        return dark_templar
    if str(id) == '77':
        return sentry
    if str(id) == '78':
        return phoenix
    if str(id) == '79':
        return carrier
    if str(id) == '80':
        return void_ray
    if str(id) == '82':
        return observer
    if str(id) == '83':
        return immortal
    if str(id) == '84':
        return probe
    if str(id) == '141':
        return archon
    if str(id) == '311':
        return adept
    if str(id) == '694':
        return disruptor
    ###zerg ids
    if str(id) == '9':
        return baneling
    if str(id) == '12' or str(id) == '13' or str(id) == '15' or str(id) == '17':
        return changeling
    if str(id) == '104':
        return drone
    if str(id) == '105':
        return zergling
    if str(id) == '106':
        return overlord
    if str(id) == '107':
        return hydralisk
    if str(id) == '108':
        return mutalisk
    if str(id) == '109':
        return ultralisk
    if str(id) == '110':
        return roach
    if str(id) == '111':
        return infestor
    if str(id) == '112':
        return corruptor
    if str(id) == '114':
        return brood_lord
    if str(id) == '126':
        return queen
    if str(id) == '129':
        return overseer
    if str(id) == '289':
        return broodling
    if str(id) == '499':
        return viper
if __name__ == "__main__":
    main()

