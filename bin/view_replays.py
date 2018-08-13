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

import sys
import os

from lib.config import *
from bin.util import *

def main(unused_args):
    ### parsed_basic or replays
    type = 'parsed_basic'
    version = '1_3a'
    replay = 'SC2CombatGenerator07-08-2018_11-55-06.SC2Replay'
    
    directory = os.path.join(REPO_DIR, type, version, replay)
    
    play_replay(directory, version)
    
def entry_point():
    app.run(main)


if __name__ == '__main__':
    app.run(main)
