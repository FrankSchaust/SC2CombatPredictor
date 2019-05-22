#!/usr/bin/env python3
# Copyright 2017 Frank Schaust and Lukas Schmelzeisen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
##     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import gzip

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

            print('Going to parse replay "{}".'.format(replay_file),
                file=sys.stderr)

            REPLAYS_SINGLE = os.path.join(REPLAYS_PARSED_DIR, version+'_2', os.path.relpath(replay_file, os.path.join(REPLAY_DIR, version)).replace('.SC2Replay', ''))
            LOG_SINGLE = os.path.join(REPO_DIR, 'log', version+'_2', os.path.relpath(replay_file, os.path.join(REPLAY_DIR, version)).replace('.SC2Replay', ''))
            os.makedirs(REPLAYS_SINGLE, exist_ok=True)
            os.makedirs(LOG_SINGLE, exist_ok=True)
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

                    #save unit and win values from last round
                    last_num_units = curr_num_units
                    last_num_wins_minerals = curr_num_wins_minerals
                    last_num_wins_vespene = curr_num_wins_vespene
                    #get values from active round
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
                        round_num += 1
                        print('Parsing Round {}...'.format(round_num),
                                file=sys.stderr)
                        assert curr_battle is None

                        curr_battle = obs.observation
                        unit_array = obs.observation.raw_data.units

                        units = []
                        units.append([])
                        units.append([])

                        for unit in unit_array:
                            if unit.owner == 1:
                                units[0].append(unit.unit_type)
                            elif unit.owner == 2:
                                units[1].append(unit.unit_type)

                    outcome = None
                    if (curr_num_wins_minerals > last_num_wins_minerals) and (curr_num_wins_vespene > last_num_wins_vespene):
                        units.append([2])
                        outcome = [0, 0, 1]
                    elif (curr_num_wins_minerals > last_num_wins_minerals) \
                        and (last_num_wins_minerals != -1):
                        units.append([0])
                        outcome = [1, 0, 0]
                    elif (curr_num_wins_vespene > last_num_wins_vespene) \
                        and (last_num_wins_vespene != -1):
                        units.append([1])                                   
                        outcome = [0, 1, 0]
                    if outcome is not None:
                        if curr_battle is not None:
                            parsed_replay_path = os.path.join(REPO_DIR, 'Proc_Data_', version, os.path.relpath(replay_file, os.path.join(REPLAY_DIR, version)).replace('.SC2Replay', ''), os.path.relpath(replay_file, os.path.join(REPLAY_DIR, version)).replace('.SC2Replay', '') + '_' + str(round_num))
                            print('Round %s parsed and saved as: %s' % (round_num, parsed_replay_path), file=sys.stderr)
                            player_relative = curr_battle.feature_layer_data.renders.player_relative
                            height_map = curr_battle.feature_layer_data.renders.height_map
                            visibility = curr_battle.feature_layer_data.renders.visibility_map
                            creep = curr_battle.feature_layer_data.renders.creep
                            power = curr_battle.feature_layer_data.renders.power
                            player_id = curr_battle.feature_layer_data.renders.player_id
                            unit_type = curr_battle.feature_layer_data.renders.unit_type
                            selected = curr_battle.feature_layer_data.renders.selected
                            hit_points = curr_battle.feature_layer_data.renders.unit_hit_points
                            energy = curr_battle.feature_layer_data.renders.unit_energy
                            shields = curr_battle.feature_layer_data.renders.unit_shields
                            unit_density = curr_battle.feature_layer_data.renders.unit_density
                            unit_density_aa = curr_battle.feature_layer_data.renders.unit_density_aa
                            feature_layers = []
                            # # computing feature layers in a Nx84x84 Tensor
                            # # Player Relative
                            data_pr = np.frombuffer(player_relative.data, dtype=np.int32) \
                                .reshape((1, 42, 42))
                            data_processed = data_pr[0].copy()
                            data_processed = np.resize(data_processed,(84, 84))
                            feature_layers.append(data_processed)
                            # # height map			
                            data_hm = np.frombuffer(height_map.data, dtype=np.int32) \
                                .reshape((1, 42, 42))
                            data_processed = data_hm[0].copy()
                            data_processed = np.resize(data_processed,(84, 84))
                            feature_layers.append(data_processed)
                            # # visibility
                            data_vi = np.frombuffer(visibility.data, dtype=np.int32) \
                                .reshape((1, 42, 42))
                            data_processed = data_vi[0].copy()
                            data_processed = np.resize(data_processed,(84, 84))
                            feature_layers.append(data_processed)
                            # # creep
                            data_cr = np.frombuffer(creep.data, dtype=np.int8) \
                                .reshape((1, -1, 2))
                            data_processed = data_cr[0].copy()
                            data_processed = np.resize(data_processed,(84, 84))
                            feature_layers.append(data_processed)
                            # # power
                            data_pw = np.frombuffer(power.data, dtype=np.int8) \
                                .reshape((1, -1, 2))
                            data_processed = data_pw[0].copy()
                            data_processed = np.resize(data_processed,(84, 84))
                            feature_layers.append(data_processed)
                            # player id 
                            data_id = np.frombuffer(player_id.data, dtype=np.int32) \
                                .reshape((1, 42, 42))
                            data_processed = data_id[0].copy()
                            data_processed = np.resize(data_processed,(84, 84))
                            feature_layers.append(data_processed)
                            # unit_type
                            data_ut = np.frombuffer(unit_type.data, dtype=np.int32) \
                                .reshape((1, 84, 84))
                            feature_layers.append(data_ut[0])
                            # # # selected
                            data_se = np.frombuffer(selected.data, dtype=np.int8) \
                                .reshape((1, -1, 2))
                            data_processed = data_se[0].copy()
                            data_processed = np.resize(data_processed,(84, 84))
                            feature_layers.append(data_processed)
                            # hit points
                            data_hp = np.frombuffer(hit_points.data, dtype=np.int32) \
                                .reshape((1, 84, 84))
                            feature_layers.append(data_hp[0])
                            # # energy
                            data_en = np.frombuffer(energy.data, dtype=np.int32) \
                                .reshape((1, 84, 84))
                            feature_layers.append(data_en[0])
                            # # shields
                            data_sh = np.frombuffer(shields.data, dtype=np.int32) \
                                .reshape((1, 84, 84))
                            feature_layers.append(data_sh[0])
                            # # unit density
                            data_de = np.frombuffer(unit_density.data, dtype=np.int32) \
                                .reshape((1, 42, 42))
                            data_processed = data_de[0].copy()
                            data_processed = np.resize(data_processed,(84, 84))
                            feature_layers.append(data_processed)
                            # # unit density aa 
                            data_da = np.frombuffer(unit_density_aa.data, dtype=np.int32) \
                                .reshape((1, 42, 42))
                            data_processed = data_da[0].copy()
                            data_processed = np.resize(data_processed,(84, 84))
                            feature_layers.append(data_processed)
                            layers = []
                            for layer in feature_layers:
                                layer = np.reshape(layer, (84, 84))
                                layers.append(layer)
                            layers = np.reshape(layers, (13*84, 84))
                            layers = np.append(layers, outcome)
                            data = pd.DataFrame(layers)
                            os.makedirs(parsed_replay_path, exist_ok=True)
                            data.to_csv(os.path.join(parsed_replay_path, 'Layer.csv'), header=False, index=False, compression='gzip')

                            units = np.array(units)
                            df = pd.DataFrame(units)
                            df.to_csv(os.path.join(LOG_SINGLE, os.path.relpath(replay_file, os.path.join(REPLAY_DIR, version)).replace('.SC2Replay', '_' + str(round_num) + '.csv')))
                            units = []                              
                            curr_battle = None
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
            os.makedirs(os.path.join(REPO_DIR, 'parsed_basic', version), exist_ok=True)
            os.rename(replay_file, os.path.join(REPO_DIR, "parsed_basic", version, os.path.relpath(replay_file, os.path.join(REPLAY_DIR, version))))
            print('Done.', file=sys.stderr)
       

def entry_point():
    app.run(main)


if __name__ == '__main__':
    app.run(main)