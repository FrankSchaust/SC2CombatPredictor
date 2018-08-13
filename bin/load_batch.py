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
import time
import gzip
import pprint

import keras
import numpy as np
import tensorflow as tf

from absl import app

from lib.config import REPLAYS_PARSED_DIR
from data import simulation_pb2

def load_batch(replay_parsed_files, capped_batch = 1000, run = 0, lastindex = 0, train = True, skip_remis = False):			
        # Collect input and output
        # data.
    xs = []
    ys = []

    i = 0
    n = lastindex
    to_cap = capped_batch
    while i < to_cap:
        simulation = simulation_pb2.Simulation()
        if n >= len(replay_parsed_files):
            #print("%s files loaded" % (i))
            break;
        with gzip.open(replay_parsed_files[n], 'rb') as file:
            #print ("Printing file number %s replay name %s" % (i, replay_parsed_files[i]))
            simulation.ParseFromString(file.read())

		
        for battle in simulation.battle:
            obs = battle.initial_observation
            feature_layers = []
			#Objects contained in feature_layer_data.renders:
			## effects 84x84 8Bits/px
			## unit_shields_ratio 
			## unit_energy_ratio
			## unit_hit_points_ratio
			## unit_density
			## unit_density_aa
			## player_relative 84x84
            ## unit_shields
			## unit_energy
			## unit_hit_points 168x168
			## selected
			## unit_type
			## player_id
			## power
			## creep
			## visibility_map
			## height_map
            #print(obs.feature_layer_data.renders)
            
			# getting the feature layers from the observed data from map
            player_relative = obs.feature_layer_data.renders.player_relative
            height_map = obs.feature_layer_data.renders.height_map
            visibility = obs.feature_layer_data.renders.visibility_map
            creep = obs.feature_layer_data.renders.creep
            power = obs.feature_layer_data.renders.power
            player_id = obs.feature_layer_data.renders.player_id
            unit_type = obs.feature_layer_data.renders.unit_type
            selected = obs.feature_layer_data.renders.selected
            hit_points = obs.feature_layer_data.renders.unit_hit_points
            energy = obs.feature_layer_data.renders.unit_energy
            shields = obs.feature_layer_data.renders.unit_shields
            unit_density = obs.feature_layer_data.renders.unit_density
            unit_density_aa = obs.feature_layer_data.renders.unit_density_aa
			
			# getting the feature layers from the observed data from the minimap
            mm_height_map = obs.feature_layer_data.minimap_renders.height_map
            mm_player_relative = obs.feature_layer_data.minimap_renders.player_relative
            mm_visibility = obs.feature_layer_data.minimap_renders.visibility_map
            mm_creep = obs.feature_layer_data.minimap_renders.creep
            mm_camera = obs.feature_layer_data.minimap_renders.camera
            mm_player_id = obs.feature_layer_data.minimap_renders.player_id
            mm_selected = obs.feature_layer_data.minimap_renders.selected
            
			
			# # ä computing feature layers in a Nx84x84 Tensor
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
			
			# # # # Appending the feature layers from the minimap
            # # # height map
            # data_hm_mm = np.frombuffer(mm_height_map.data, dtype=np.int32) \
                # .reshape((1, 32, 32))
            # data_processed = data_hm_mm[0].copy()
            # data_processed = np.resize(data_processed,(84, 84))
            # feature_layers.append(data_processed)
			
			# # # visibility map
            # data_vi_mm = np.frombuffer(mm_visibility.data, dtype=np.int32) \
                # .reshape((1, 32, 32))
            # data_processed = data_vi_mm[0].copy()
            # data_processed = np.resize(data_processed,(84, 84))
            # feature_layers.append(data_processed)
			
			# # # creep
            # data_cr_mm = np.frombuffer(mm_height_map.data, dtype=np.int8) \
                # .reshape((1, -1, 2))
            # data_processed = data_cr_mm[0].copy()
            # data_processed = np.resize(data_processed,(84, 84))
            # feature_layers.append(data_processed)
			
			# # # camera 
            # data_ca_mm = np.frombuffer(mm_height_map.data, dtype=np.int32) \
                # .reshape((1, 32, 32))
            # data_processed = data_ca_mm[0].copy()
            # data_processed = np.resize(data_processed,(84, 84))
            # feature_layers.append(data_processed)			
			
	        # # # player id
            # data_id_mm = np.frombuffer(mm_height_map.data, dtype=np.int32) \
                # .reshape((1, 32, 32))
            # data_processed = data_id_mm[0].copy()
            # data_processed = np.resize(data_processed,(84, 84))
            # feature_layers.append(data_processed)
			
			# # # player relative
            # data_pr_mm = np.frombuffer(mm_height_map.data, dtype=np.int32) \
                # .reshape((1, 32, 32))
            # data_processed = data_pr_mm[0].copy()
            # data_processed = np.resize(data_processed,(84, 84))
            # feature_layers.append(data_processed)
			
			# # # selected
            # data_se_mm = np.frombuffer(mm_height_map.data, dtype=np.int8) \
                # .reshape((1, -1, 2))
            # data_processed = data_se_mm[0].copy()
            # data_processed = np.resize(data_processed,(84, 84))
            # feature_layers.append(data_processed)
			
			# # Make list a np array
            feature_layers = np.array(feature_layers) 
            #print("Feature layer length: %s, feature layer[1]: %s, %s" % (len(feature_layers), len(feature_layers[17]), len(feature_layers[17][0])))
            
            if skip_remis: 
                if battle.outcome != 2:
                    xs.append(feature_layers)
                    ys.append(battle.outcome)
                    i += 1 
                #print(i,n)
            if not skip_remis:
                xs.append(feature_layers)
                ys.append(battle.outcome)
                i += 1
        n += 1
    lastindex = n
    #print(i,n, lastindex)
		
    xs = np.array(xs)
    ys = np.array(ys)
    #print(len(xs), len(ys))
    split = int(len(xs)*0.1)
    # # Make train / test split
    xs_train = xs[:-split]
    ys_train = ys[:-split]
    xs_test = xs[-split:]
    ys_test = ys[-split:]
    #print(xs_train.size, xs_train.shape, xs_train[0].size, xs_train[0].shape)
    # # Convert labels to one-hot.
    if skip_remis:
        num_classes = 2
    else:    
        num_classes = 3
    ys_train = keras.utils.to_categorical(ys_train, num_classes=num_classes)
    ys_test = keras.utils.to_categorical(ys_test, num_classes=num_classes)
    #shaping
    depth = 13
    img_rows, img_cols = 84, 84
    input_shape = (depth, img_rows,img_cols,1)
	
    xs_train = xs_train.astype(np.float32)
    xs_test = xs_test.astype(np.float32)
    
    #pp = pprint.PrettyPrinter(width=84)
    #pp.pprint(xs_test[0])
    #pp.pprint(xs_test[0][0])
	
    # # Length check
    assert(len(xs_train) == len(ys_train))
    assert(len(xs_test) == len(ys_test))
        
    xs_train = np.frombuffer(xs_train, dtype=np.float32).reshape(xs_train.shape[0], depth, 84, 84, 1)
    xs_test = np.frombuffer(xs_test, dtype=np.float32).reshape(xs_test.shape[0],depth, 84, 84, 1)
    #print("xs_train Format: %s, xs_test Format %s" % (xs_train.shape, xs_test.shape))
    if(train):
        return xs_train, xs_test, ys_train, ys_test, lastindex    
    else:
        return xs_test, ys_test, lastindex
