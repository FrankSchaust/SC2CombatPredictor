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

from lib.config import SCREEN_RESOLUTION, MINIMAP_RESOLUTION, MAP_PATH, \
    REPLAYS_PARSED_DIR, REPLAY_DIR, REPO_DIR, STANDARD_VERSION
from data.simulation_pb2 import Battle, Simulation


def main():
    version = '1_3c'
    replay_files = []
    for root, dir, files in os.walk(os.path.join(REPLAY_DIR, version)):
        for file in files:
            if file.endswith(".SC2Replay"):
                replay_files.append(os.path.join(root, file))
        for replay_file in replay_files:
            print('Going to parse replay "{}".'.format(replay_file),
              file=sys.stderr)
            REPLAYS_SINGLE = os.path.join(REPLAYS_PARSED_DIR, version, os.path.relpath(replay_file, os.path.join(REPLAY_DIR, version)).replace('.SC2Replay', ''))
            LOG_SINGLE = os.path.join(REPO_DIR, 'log', version, os.path.relpath(replay_file, os.path.join(REPLAY_DIR, version)).replace('.SC2Replay', ''))
            SCREENS_SINGLE = os.path.join(REPO_DIR, 'screens', version, os.path.relpath(replay_file, os.path.join(REPLAY_DIR, version)).replace('.SC2Replay', ''))
            os.makedirs(REPLAYS_SINGLE, exist_ok=True)
            os.makedirs(LOG_SINGLE, exist_ok=True)
            os.makedirs(SCREENS_SINGLE, exist_ok=True)
            simulation = Simulation()
            run_config = run_configs.get()
            PATH = MAP_PATH + '-v' + version +'.SC2Map'
            replay_data = run_config.replay_data(replay_file)
            with run_config.start(game_version=get_replay_version(replay_data),
                              full_screen=False) as controller:
                print('Starting replay...', file=sys.stderr)
                info = controller.replay_info(replay_data=replay_data)
                time = info.game_duration_seconds
                if time > 4000:
                    os.remove(replay_file)
                    print('removed %s because game time exceeded 1 hour (actual replay duration was %.2f)'  % (replay_file, time))
                else: 
                    print('File okay, Duration is %.2f' % (time))
                    
                    
                    
if __name__ == "__main__":
    import sys
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    main()