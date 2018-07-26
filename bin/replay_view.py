#!/usr/bin/env python3
# Copyright 2017 Lukas Schmelzeisen. All Rights Reserved.
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

from absl import app
from pysc2 import run_configs
from pysc2.bin.play import get_replay_version
from s2clientprotocol.common_pb2 import Size2DI
from s2clientprotocol.sc2api_pb2 import InterfaceOptions, RequestStartReplay, \
    SpatialCameraSetup

from lib.config import SCREEN_RESOLUTION, MINIMAP_RESOLUTION, MAP_PATH, \
    REPLAYS_PARSED_DIR, REPLAY_DIR, REPO_DIR
from data.simulation_pb2 import Battle, Simulation


def main(unused_argv):
    replay_files = []
    for root, dir, files in os.walk(REPLAY_DIR):
        for file in files:
            if file.endswith(".SC2Replay"):
                replay_files.append(os.path.join(root, file))

    for replay_file in replay_files:
        print('Going to parse replay "{}".'.format(replay_file),
              file=sys.stderr)

        simulation = Simulation()
		
        run_config = run_configs.get()
        replay_data = run_config.replay_data(replay_file)
        with run_config.start(game_version=get_replay_version(replay_data),
                              full_screen=False) as controller:
            print('Starting replay...', file=sys.stderr)
            controller.start_replay(RequestStartReplay(
                replay_data=replay_data,
                # Overwrite map_data because we know where our map is located
                # (so we don't care about the map path stored in the replay).
                map_data=run_config.map_data(MAP_PATH),
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

            round_num = 0

            obs = controller.observe()
            while not obs.player_result:
                # We advance 16 steps at a time here because this is the largest
                # value which still ensures that we will have at least one
                # observation between different phases (units spawn, units
                # engage, outcome determined, units remove) when wait time is
                # set to 1.0 game seconds in editor.
                controller.step(1)
                obs = controller.observe()
                time.sleep(0.01)
                
                

        
        print('Done.')


def entry_point():
    app.run(main)


if __name__ == '__main__':
    app.run(main)
