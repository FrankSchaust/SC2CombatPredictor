#!/usr/bin/env python3
# Copyright 2017 Frank Schaust and Lukas Schmelzeisen. All Rights Reserved.
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
from mss import mss

import pandas as pd
import numpy as np
from absl import app
from pysc2 import run_configs
from pysc2.bin.play import get_replay_version
from s2clientprotocol.common_pb2 import Size2DI
from s2clientprotocol.sc2api_pb2 import InterfaceOptions, RequestStartReplay, \
    SpatialCameraSetup

from lib.config import MAP_PATH, \
    REPLAYS_PARSED_DIR, REPLAY_DIR, REPO_DIR, STANDARD_VERSION, REPLAY_VERSION
from data.simulation_pb2 import Battle, Simulation


def main(unused_argv, version=STANDARD_VERSION):
    version = REPLAY_VERSION
    replay_files = []
    for root, dir, files in os.walk(os.path.join(REPLAY_DIR, version)):
        for file in files:
            if file.endswith(".SC2Replay"):
                replay_files.append(os.path.join(root, file))
        for replay_file in replay_files:
            try:
                print('Going to parse replay "{}".'.format(replay_file),
                  file=sys.stderr)


                simulation = Simulation()
                REPLAYS_SINGLE = os.path.join(REPLAYS_PARSED_DIR, version+'_2', os.path.relpath(replay_file, os.path.join(REPLAY_DIR, version)).replace('.SC2Replay', ''))
                LOG_SINGLE = os.path.join(REPO_DIR, 'log', version+'_2', os.path.relpath(replay_file, os.path.join(REPLAY_DIR, version)).replace('.SC2Replay', ''))
                SCREENS_SINGLE = os.path.join(REPO_DIR, 'screens', version, os.path.relpath(replay_file, os.path.join(REPLAY_DIR, version)).replace('.SC2Replay', ''))
                os.makedirs(REPLAYS_SINGLE, exist_ok=True)
                os.makedirs(LOG_SINGLE, exist_ok=True)
                os.makedirs(SCREENS_SINGLE, exist_ok=True)
                run_config = run_configs.get()
                PATH = MAP_PATH + '-v' + version +'.SC2Map'
                replay_data = run_config.replay_data(replay_file)
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
                    info = controller.replay_info(replay_data=replay_data)
                    time = info.game_duration_seconds
                    if time > 6000:
                        os.remove(replay_file)
                        print('removed %s because game time exceeded 1 hour (actual replay duration was %.2f)'  % (replay_file, time))
                        break
                    else: 
                        print('File okay, Duration is %.2f' % (time))
                    round_num = 0
                    curr_battle = None
                    last_num_units, curr_num_units = -1, -1
                    last_num_wins_minerals, curr_num_wins_minerals = -1, -1
                    last_num_wins_vespene, curr_num_wins_vespene = -1, -1
                    units = []
                    obs = controller.observe()
                    while not obs.player_result:
                        # We advance 16 steps at a time here because this is the largest
                        # value which still ensures that we will have at least one
                        # observation between different phases (units spawn, units
                        # engage, outcome determined, units remove) when wait time is
                        # set to 1.0 game seconds in editor.
                        controller.step(16)
                        obs = controller.observe()

                        last_num_units = curr_num_units
                        last_num_wins_minerals = curr_num_wins_minerals
                        last_num_wins_vespene = curr_num_wins_vespene

                        curr_num_units = len(obs.observation.raw_data.units)
                        curr_num_wins_minerals = \
                            obs.observation.player_common.minerals
                        curr_num_wins_vespene = \
                            obs.observation.player_common.vespene

                        if curr_num_units != 0 and last_num_units == 0 and curr_battle is None:
                            broodling = False
                            units = obs.observation.raw_data.units
                            for unit in units:
                                if unit.unit_type == 289:
                                    broodling = True
                            if len(obs.observation.raw_data.units) < 12 and broodling:
                                curr_num_units = 0
                                curr_battle = None
                                continue
                            round_num += 1
                            print('Parsing Round {}...'.format(round_num),
                                  file=sys.stderr)
                            assert curr_battle is None
                            
                            simulation = Simulation()
                            curr_battle = simulation.battle.add()
                            curr_battle.replay_file = replay_file
                            curr_battle.round_num = round_num
                            curr_battle.initial_observation.CopyFrom(obs.observation)
                            unit_array = obs.observation.raw_data.units
                            units = []
                            units.append([])
                            units.append([])
                            for unit in unit_array:
                                if unit.owner == 1:
                                    units[0].append(unit.unit_type)
                                elif unit.owner == 2:
                                    units[1].append(unit.unit_type)
                            # print(units)
                            # with mss() as sct:
                                # filename = sct.shot(mon=2, output=os.path.join('screens', 
                                                                                # os.path.relpath(replay_file, REPLAY_DIR).replace('.SC2Replay', ''), 
                                                                                # os.path.relpath(replay_file, REPLAY_DIR).replace('.SC2Replay', str(round_num))))
                                # print(filename)
                        outcome = None
                        if (curr_num_wins_minerals > last_num_wins_minerals) and (curr_num_wins_vespene > last_num_wins_vespene):
                            outcome = Battle.REMIS
                            units.append([2])
                        elif (curr_num_wins_minerals > last_num_wins_minerals) \
                            and (last_num_wins_minerals != -1):
                            outcome = Battle.TEAM_MINERALS_WON
                            units.append([0])
                        elif (curr_num_wins_vespene > last_num_wins_vespene) \
                            and (last_num_wins_vespene != -1):
                            outcome = Battle.TEAM_VESPENE_WON
                            units.append([1])
                        if outcome is not None:
                            if curr_battle is not None:
                                # print('Going to remove corrupted replay "{}".'.format(replay_file))
                                # os.remove(replay_file)
                                # error_flag=True
                                # break;
                                curr_battle.outcome = outcome
                                string = '_' + str(round_num) + '.SC2Replay_parsed.gz'
                                replay_parsed_file = os.path.join(
                                REPLAYS_SINGLE,
                                os.path.relpath(replay_file, os.path.join(REPLAY_DIR, version+'_2')).replace(
                                '.SC2Replay', string))
                                print('Round %s parsed and saved as: %s' % (round_num, replay_parsed_file), file=sys.stderr)
                                os.makedirs(REPLAYS_SINGLE, exist_ok=True)
                                with gzip.open(replay_parsed_file, 'wb') as file:
                                    file.write(simulation.SerializeToString())
                                
                                units = np.array(units)
                                df = pd.DataFrame(units)
                                
                                df.to_csv(os.path.join(LOG_SINGLE, os.path.relpath(replay_file, os.path.join(REPLAY_DIR, version+'_2')).replace('.SC2Replay', '_' + str(round_num) + '.csv')))
                                units = []                              
                                curr_battle = None
            except ValueError:
                print("ValueError going for next replay")
            except AssertionError:
                print('AssertionError going for next replay')
            except KeyboardInterrupt:
                print('Process finished')
                break;
            except:
                print('Generell Error going for next replay')
                # if (error_flag == False):
                    # replay_parsed_file = os.path.join(
                        # REPLAYS_PARSED_DIR,
                        # os.path.relpath(replay_file, REPLAY_DIR).replace(
                            # '.SC2Replay', '.SC2Replay_parsed'))
                    # print('Replay completed. Saving parsed replay to "{}".'
                          # .format(replay_parsed_file), file=sys.stderr)
                    # os.makedirs(REPLAYS_PARSED_DIR, exist_ok=True)
                    # with open(replay_parsed_file, 'wb') as file:
                        # file.write(simulation.SerializeToString())
                    # os.rename(replay_file, os.path.join(REPO_DIR, "parsed_basic", os.path.relpath(replay_file, REPLAY_DIR)))
            os.makedirs(os.path.join(REPO_DIR, 'parsed_basic', version+'_2'), exist_ok=True)
            os.rename(replay_file, os.path.join(REPO_DIR, "parsed_basic", version, os.path.relpath(replay_file, os.path.join(REPLAY_DIR, version))))
            print('Done.', file=sys.stderr)
       


def entry_point():
    app.run(main)


if __name__ == '__main__':
    app.run(main)
