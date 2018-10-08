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
import datetime
import time

from absl import app
from pysc2 import run_configs
from s2clientprotocol.sc2api_pb2 import Computer, LocalMap, InterfaceOptions, \
    Participant, PlayerSetup, RequestCreateGame, RequestJoinGame, VeryEasy
from s2clientprotocol.common_pb2 import Random


from lib.config import MAP_PATH, REPLAY_DIR, STANDARD_VERSION


def main(ununsed_argv):	
    demo_counter = 0
    fail_counter = 0
    while 1:
        try: 
            game_routine(version='1_3b')
            time.sleep(5)
            demo_counter += 1
        except RuntimeError:
            print("Observer crashed, demo invalid")
            fail_counter += 1
        except KeyboardInterrupt:
            print("%s demos genrated this run, %s demos were unusable" % (demo_counter, fail_counter))
            break;
        except ConnectionResetError:
            print("Unexpected Connection error, demo invalid")
            fail_counter += 1
        except ConnectionError:
            print("Unexpected Connection error, demo invalid")
            fail_counter += 1
        except:
            print("Unexpected Connection error, demo invalid")
            fail_counter += 1
def game_routine(version = STANDARD_VERSION):
        PATH = MAP_PATH + '-v' + version +'.SC2Map'
        run_config = run_configs.get()
        with run_config.start(full_screen=False) as controller:		
            print('Starting map "{}"...'.format(PATH), file=sys.stderr)
            controller.create_game(RequestCreateGame(
                local_map=LocalMap(
                    map_path=PATH,
                    map_data=run_config.map_data(PATH)),
                player_setup=[
                    PlayerSetup(type=Computer, race=Random, difficulty=VeryEasy),
                    PlayerSetup(type=Computer, race=Random, difficulty=VeryEasy),
                    PlayerSetup(type=Participant),
                ],
                realtime=False))

            print('Joining game...', file=sys.stderr)
            controller.join_game(RequestJoinGame(
                race=Random,
                # We just want to save the replay, so no special observations.
                options=InterfaceOptions()))

            print('Stepping through game...', file=sys.stderr)
            print('Remember that you have to manually configure the map settings '
                  'ingame for now!', file=sys.stderr)

            obs = controller.observe()
            while not obs.player_result:
                # We just want to save the replay and don't care about observations,
                # so we use a large step_size for speed. But don't make it to large
                # because we need user interaction at the start.
                controller.step(8)
                obs = controller.observe()
            replay_path = os.path.join(REPLAY_DIR, version, ('SC2CombatGenerator' + datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S') + '.SC2Replay'))
            os.makedirs(os.path.join(REPLAY_DIR, version), exist_ok=True)
            print('Game completed. Saving replay to "{}".'.format(replay_path),
                  file=sys.stderr)
            
            with open(replay_path, 'wb') as file:
                file.write(controller.save_replay())
            print('Done.', file=sys.stderr)

def entry_point():
    app.run(main)


if __name__ == '__main__':
    app.run(main)
