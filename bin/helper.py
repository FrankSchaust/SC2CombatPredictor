import os
import sys
import gzip
from mss import mss

import pandas as pd
import numpy as np
import tensorflow as tf
from absl import app
from pysc2 import run_configs
from pysc2.bin.play import get_replay_version
from s2clientprotocol.common_pb2 import Size2DI
from s2clientprotocol.sc2api_pb2 import InterfaceOptions, RequestStartReplay, \
    SpatialCameraSetup

from bin.util import *
from bin.load_data_pipeline import *
from lib.config import SCREEN_RESOLUTION, MINIMAP_RESOLUTION, MAP_PATH, \
    REPLAYS_PARSED_DIR, REPLAY_DIR, REPO_DIR, STANDARD_VERSION
from data.simulation_pb2 import Battle, Simulation


def main():
    versions = ['1_3d']  


    #Loading example files
    replay_parsed_files = []
    replay_parsed_files = build_file_array(version=versions)
    for file in replay_parsed_files:
        LOG_SINGLE = os.path.join(REPO_DIR, 'Proc_Data_', versions[0], os.path.relpath(file, os.path.join(REPLAYS_PARSED_DIR, versions[0])).replace('.SC2Replay_parsed.gz', ''))
        os.makedirs(LOG_SINGLE, exist_ok=True)
        #print(file)
        images = load_batch([file])
        layers = []
        for layer in images[0][0]:
            layer = np.reshape(layer,(84,84))
            layers.append(layer)
        layers = np.reshape(layers, (13*84, 84))
        layers = np.append(layers, images[1][0])
        print(images[1][0])
        data = pd.DataFrame(layers)
        data.to_csv(os.path.join(LOG_SINGLE, 'Layer.csv'), header=False, index=False, compression='gzip')
        #print(images[0])


if __name__ == "__main__":
    import sys
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    main()