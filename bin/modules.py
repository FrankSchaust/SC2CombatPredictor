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

import tensorflow as tf

from bin.util import print_layer_details

def Max_Pooling(x, pool_size=[1,3,3], stride=(1,2,2), padding='VALID'):
    return tf.layers.max_pooling3d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)
def Avg_Pooling(x, pool_size=[1,3,3], stride=(1,1,1), padding='SAME'):    
    return tf.layers.average_pooling3d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)
def Conv_Layer(input, filter, kernel, stride=(1,1,1), padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        return tf.layers.conv3d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride, padding=padding, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
def BN_Conv_Layer(input, filter, kernel, stride=(1,1,1), padding='SAME', layer_name="conv"):
        x = Conv_Layer(input, filter, kernel, stride, padding, layer_name)
        return tf.layers.batch_normalization(inputs=x, training=True)
        
def Concat(x): 
    return tf.concat(x, 4)


# definitions for inception v4
def stem(input, scope, base=8):

    with tf.name_scope(scope):
        x_ = Conv_Layer(input, filter=base, kernel=[1,3,3], layer_name=scope+'_conv1')
        x_ = Conv_Layer(x_, filter=base, kernel=[1,3,3], padding='VALID', layer_name=scope+'_conv2')
        x = Conv_Layer(x_, filter=base*2, kernel=[1,3,3], layer_name=scope+'_conv3')
        # max_x_1 = Max_Pooling(x)
        # conv_x_1 = Conv_Layer(x, filter=96, kernel=[1,3,3], stride=(1,2,2), padding='VALID', layer_name=scope+'_conv4')
        
        # x = Concat([max_x_1, conv_x_1])
        split_x_1 = Conv_Layer(x, filter=base*2, kernel=[1,1,1], layer_name=scope+'_split_conv1')
        split_x_1 = Conv_Layer(split_x_1, filter=base*3, kernel=[1,3,3], padding='VALID', layer_name=scope+'_split_conv2')
        
        split_x_2 = Conv_Layer(x, filter=base*2, kernel=[1,1,1], layer_name=scope+'_split_conv3')
        split_x_2 = Conv_Layer(split_x_2, filter=base*2, kernel=[1,7,1], layer_name=scope+'_split_conv4')
        split_x_2 = Conv_Layer(split_x_2, filter=base*2, kernel=[1,1,7], layer_name=scope+'_split_conv5')
        split_x_2 = Conv_Layer(split_x_2, filter=base*3, kernel=[1,3,3], padding='VALID', layer_name=scope+'_split_conv6')
        
        x = Concat([split_x_1, split_x_2])
        # split_conv_x = Conv_Layer(x, filter=192, kernel=[1,3,3], stride=(1,2,2), padding='VALID', layer_name=scope+'_conv5')
        # split_max_x = Max_Pooling(x)
        
        # x = Concat([split_conv_x, split_max_x])
        x = tf.layers.batch_normalization(inputs=x, training=True)
        return x

def inception_a( input, scope, base=8):
    with tf.name_scope(scope):
        # far left side
        x_fl = Avg_Pooling(input)
        x_fl = Conv_Layer(x_fl, filter=base*3, kernel=[1,1,1], layer_name=scope+"_split_conv1")
        # close left side
        x_cl = Conv_Layer(input, filter=base*3, kernel=[1,1,1], layer_name=scope+"_split_conv2")
        # close right side
        x_cr = Conv_Layer(input, filter=base*2, kernel=[1,1,1], layer_name=scope+"_split_conv3")
        x_cr = Conv_Layer(x_cr, filter=base*3, kernel=[1,3,3], layer_name=scope+"_split_conv4")
        #far right side
        x_fr = Conv_Layer(input, filter=base*2, kernel=[1,1,1], layer_name=scope+"_split_conv5")
        x_fr = Conv_Layer(x_fr, filter=base*3, kernel=[1,3,3], layer_name=scope+"_split_conv6")
        x_fr = Conv_Layer(x_fr, filter=base*3, kernel=[1,3,3], layer_name=scope+"_split_conv7")
        
        x = Concat([x_fl, x_cl, x_cr, x_fr])
        x = tf.layers.batch_normalization(inputs=x, training=True)
        return x
        
def inception_b( input, scope, base=8):
    with tf.name_scope(scope):
        # far left side
        x_fl = Avg_Pooling(input)
        x_fl = Conv_Layer(x_fl, filter=base*4, kernel=[1,1,1], layer_name=scope+"_split_conv1")
        # close left side
        x_cl = Conv_Layer(input, filter=base*12, kernel=[1,1,1], layer_name=scope+"_split_conv2")
        # close right side
        x_cr = Conv_Layer(input, filter=base*6, kernel=[1,1,1], layer_name=scope+"_split_conv3")
        x_cr = Conv_Layer(x_cr, filter=base*7, kernel=[1,1,7], layer_name=scope+"_split_conv4")
        x_cr = Conv_Layer(x_cr, filter=base*8, kernel=[1,1,7], layer_name=scope+"_split_conv5")
        #far right side
        x_fr = Conv_Layer(input, filter=base*6, kernel=[1,1,1], layer_name=scope+"_split_conv6")
        x_fr = Conv_Layer(x_fr, filter=base*6, kernel=[1,1,7], layer_name=scope+"_split_conv7")
        x_fr = Conv_Layer(x_fr, filter=base*7, kernel=[1,7,1], layer_name=scope+"_split_conv8")
        x_fr = Conv_Layer(x_fr, filter=base*7, kernel=[1,1,7], layer_name=scope+"_split_conv9")
        x_fr = Conv_Layer(x_fr, filter=base*8, kernel=[1,7,1], layer_name=scope+"_split_conv10")
        
        x = Concat([x_fl, x_cl, x_cr, x_fr])
        x = tf.layers.batch_normalization(inputs=x, training=True)
        return x        
    
def inception_c( input, scope, base=8):
    with tf.name_scope(scope):
        # far left side
        x_fl = Avg_Pooling(input)
        x_fl = Conv_Layer(x_fl, filter=base*8, kernel=[1,1,1], layer_name=scope+"_split_conv1")
        # close left side
        x_cl = Conv_Layer(input, filter=base*8, kernel=[1,1,1], layer_name=scope+"_split_conv2")
        # close right side
        x_cr = Conv_Layer(input, filter=base*12, kernel=[1,1,1], layer_name=scope+"_split_conv3")
        x_cr_1 = Conv_Layer(x_cr, filter=base*8, kernel=[1,1,3], layer_name=scope+"_split_conv4")
        x_cr_2 = Conv_Layer(x_cr, filter=base*8, kernel=[1,3,1], layer_name=scope+"_split_conv5")
        #far right side
        x_fr = Conv_Layer(input, filter=base*12, kernel=[1,1,1], layer_name=scope+"_split_conv6")
        x_fr = Conv_Layer(x_fr, filter=base*14, kernel=[1,1,3], layer_name=scope+"_split_conv7")
        x_fr = Conv_Layer(x_fr, filter=base*16, kernel=[1,3,1], layer_name=scope+"_split_conv8")
        x_fr_1 = Conv_Layer(x_fr, filter=base*8, kernel=[1,3,1], layer_name=scope+"_split_conv9")
        x_fr_2 = Conv_Layer(x_fr, filter=base*8, kernel=[1,1,3], layer_name=scope+"_split_conv10")
        
        x = Concat([x_fl, x_cl, x_cr_1, x_cr_2, x_fr_1, x_fr_2])
        x = tf.layers.batch_normalization(inputs=x, training=True)
        return x 
   
def reduction_a( input, scope, base=8):
    with tf.name_scope(scope):
        max_pool = tf.layers.max_pooling3d(input, pool_size=[1,3,3], strides=(1,2,2), padding='VALID')

        conv_1 = Conv_Layer(input, filter=base*12, kernel=[1,3,3], stride=(1,2,2), padding='VALID', layer_name=scope+"_split_conv1")
        
        conv_2 = Conv_Layer(input, filter=base*8, kernel=[1,1,1], layer_name=scope+"_split_conv2")
        conv_2 = Conv_Layer(conv_2, filter=base*8, kernel=[1,3,3], layer_name=scope+"_split_conv3")
        conv_2 = Conv_Layer(conv_2, filter=base*12, kernel=[1,3,3], stride=(1,2,2), padding='VALID', layer_name=scope+"_split_conv4")
        
        x = Concat([max_pool, conv_1, conv_2])
        x = tf.layers.batch_normalization(inputs=x, training=True)
        
        return x
   
def reduction_b( input, scope, base=8):
    with tf.name_scope(scope):
        max_pool = tf.layers.max_pooling3d(input, pool_size=[1,3,3], strides=(1,2,2), padding='VALID')

        conv_1 = Conv_Layer(input, filter=base*8, kernel=[1,1,1], layer_name=scope+"_split_conv1")
        conv_1 = Conv_Layer(conv_1, filter=base*12, kernel=[1,3,3], stride=(1,2,2), padding='VALID', layer_name=scope+"_split_conv2")
        
        conv_2 = Conv_Layer(input, filter=base*8, kernel=[1,1,1], layer_name=scope+"_split_conv3")
        conv_2 = Conv_Layer(conv_2, filter=base*9, kernel=[1,3,3], stride=(1,2,2), padding='VALID', layer_name=scope+"_split_conv4")
        
        conv_3 = Conv_Layer(input, filter=base*8, kernel=[1,1,1], layer_name=scope+"_split_conv5")
        conv_3 = Conv_Layer(conv_3, filter=base*9, kernel=[1,3,3], layer_name=scope+"_split_conv6")
        conv_3 = Conv_Layer(conv_3, filter=base*10, kernel=[1,3,3], stride=(1,2,2), padding='VALID', layer_name=scope+"_split_conv7")
        
        x = Concat([max_pool, conv_1, conv_2, conv_3])
        x = tf.layers.batch_normalization(inputs=x, training=True)
        
        return x 
        
# definition for squeeze and excitation
def se_layer(input, out_dim, ratio, scope):
    with tf.name_scope(scope):
        squeeze = tf.keras.backend.mean(input, axis=[1,2,3])
        excitation = tf.layers.dense(inputs=squeeze, units=out_dim/ratio)
        excitation = tf.nn.relu(excitation)
        excitation = tf.layers.dense(inputs=excitation, units=out_dim)
        excitation = tf.nn.sigmoid(excitation)
        
        excitation = tf.reshape(excitation, [-1, 1, 1, 1, out_dim])
        scale = input * excitation
        
        return scale 

# inception v1 module
def inception(inputs, kernels=[1,1,1,1,1,1]):
 ### TODO                
    # far left side of inception module
    inception_1x1 = tf.layers.conv3d(inputs=inputs, filters=kernels[0], kernel_size=[1, 1, 1], strides=(1,1,1), padding='same', activation=tf.nn.relu)
    # near left side
    inception_red_3x3 = tf.layers.conv3d(inputs=inputs, filters=kernels[1], kernel_size=[1, 1, 1], strides=(1,1,1), padding='same', activation=tf.nn.relu)
    inception_3x3 = tf.layers.conv3d(inputs=inception_red_3x3, filters=kernels[2], kernel_size=[1, 3, 3], strides=(1,1,1), padding='same', activation=tf.nn.relu)
    # near right side
    inception_red_5x5 = tf.layers.conv3d(inputs=inputs, filters=kernels[3], kernel_size=[1, 1, 1], strides=(1,1,1), padding='same', activation=tf.nn.relu)
    inception_5x5 = tf.layers.conv3d(inputs=inception_red_5x5, filters=kernels[4], kernel_size=[1, 5, 5], strides=(1,1,1), padding='same', activation=tf.nn.relu)
    # far right side
    inception_max_pool = tf.layers.max_pooling3d(inputs=inputs, pool_size=[1, 2, 2], strides=(1,1,1))
    inception_max_pool_1x1 = tf.layers.conv3d(inputs=inputs, filters=kernels[5], kernel_size=[1, 1, 1], strides=(1,1,1), padding='same', activation=tf.nn.relu)
    inception = tf.concat([inception_1x1, inception_3x3, inception_5x5, inception_max_pool_1x1], 4)
    inception_bn = tf.layers.batch_normalization(inputs=inception, training=True)
    return inception_bn

#TODO
# residual modules
### batch normalization along the channel axis (channels_last => 4)
def batch_norm(input):
    norm = tf.layers.batch_normalization(inputs=input, axis=4)
    return tf.nn.relu(norm)
    
### define a convolution layer followed by batch normalization
def conv_bn_relu(input, filters, kernel_size, strides, padding, kernel_regularizer):
    conv = tf.layers.conv3d(inputs=input, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_regularizer=kernel_regularizer)
    return batch_norm(conv)


def bn_relu_conv(input, filters, kernel_size, strides, padding, kernel_regularizer):
    norm = batch_norm(input)
    return tf.layers.conv3d(inputs=norm, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_regularizer=kernel_regularizer)
### define the shortcut function
def shortcut3d(input, residual, strides):
    equal_channels = residual.get_shape()[4] == input.get_shape()[4]
    shortcut = input
    if strides[1] > 1:
        shortcut = tf.layers.conv3d(
            inputs=input,
            filters=residual.get_shape()[4],
            kernel_size=[1, 3, 3],
            strides=strides,
            padding='VALID',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4))
    if not equal_channels and strides[1] == 1:
            shortcut = tf.layers.conv3d(
            inputs=input,
            filters=residual.get_shape()[4],
            kernel_size=[1, 1, 1],
            strides=strides,
            padding='VALID',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4))
    return tf.nn.relu(tf.keras.layers.add([shortcut, residual]))

### define a 3-dimensional residual block
def residual_block_3d(input, block_function, filters, repetitions, kernel_regularizer, is_first_layer=False, scope=''):
    for i in range(repetitions):
        with tf.name_scope(scope + '_' + str(i)):
            strides = (1, 1, 1)
            if i == 0 and not is_first_layer:
                strides=(1, 2, 2)
            input = block_function(input, filters=filters, strides=strides, kernel_regularizer=kernel_regularizer)   
            _ = print_layer_details(scope + '_' + str(i), input.get_shape())
    return input
    
### define basic block 
def basic_block(input, filters, strides=(1,1,1), kernel_regularizer=tf.keras.regularizers.l2(1e-4)):
    if strides[1] > 1:
        conv1 = tf.layers.conv3d(inputs=input, filters=filters, kernel_size=[1,3,3], strides=strides, padding='VALID', kernel_regularizer=kernel_regularizer)
    else:
        conv1 = tf.layers.conv3d(inputs=input, filters=filters, kernel_size=[1,3,3], strides=strides, padding='SAME', kernel_regularizer=kernel_regularizer)
        
    residual = bn_relu_conv(conv1, filters=filters, kernel_size=[1,3,3], strides=(1,1,1), padding='SAME', kernel_regularizer=kernel_regularizer)
    return shortcut3d(input, residual, strides)

 
### define a bottleneck block
def bottleneck(input, filters, strides=(1,1,1), kernel_regularizer=tf.keras.regularizers.l2(1e-4)):
        conv_1_1 = tf.layers.conv3d(inputs=input, filters=filters, kernel_size=[1,1,1], strides=strides, padding='SAME', kernel_regularizer=kernel_regularizer)
        
        conv_3_3 = tf.layers.conv3d(inputs=conv_1_1, filters=filters, kernel_size=[1,3,3], strides=(1,1,1), kernel_regularizer=kernel_regularizer)
        
        residual = bn_relu_conv(conv_3_3, filters=filters*4, kernel_size=[1,1,1], strides=(1,1,1), padding='SAME', kernel_regularizer=kernel_regularizer)
        # print(is_first_block_of_first_layer, input.get_shape(), conv_1_1.get_shape(), conv_3_3.get_shape())
        return shortcut3d(input, residual, strides)