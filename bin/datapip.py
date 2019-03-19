import os
import sys

import pandas as pd
import numpy as np
import tensorflow as tf
from absl import app
from bin.util import *
from lib.config import SCREEN_RESOLUTION, MINIMAP_RESOLUTION, MAP_PATH, \
    REPLAYS_PARSED_DIR, REPLAY_DIR, REPO_DIR, STANDARD_VERSION
from bin.all_conv_cnn import all_conv

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    versions = ['1_3d_10sup']  
    num_classes = 3
    batch_size = 20 

    file = build_file_array(type='csv', version=versions)
    dataset = tf.data.TextLineDataset(file[:5000], compression_type='GZIP')

    dataset = dataset.batch(13*84*84+3)
    dataset = dataset.map(parse_func, num_parallel_calls=4)

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(1)

    iterator = dataset.make_initializable_iterator()
    next_feature = iterator.get_next()
        

    x = next_feature[0]
    
    y_ = all_conv(x) 
    
    softmax = tf.nn.softmax(y_)

    #get trainable params
    para = get_params(tf.trainable_variables())
    print(para)
    y = next_feature[1]
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y))

    # add an optimiser
    optimiser = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cross_entropy)

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(softmax, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # setup the initialisation operator
    init_op = tf.global_variables_initializer()
    # setup the save and restore functionality for variables 
    saver = tf.train.Saver()

    with tf.Session() as sess: 
        sess.run(init_op)

        for i in range(100):
            sess.run(iterator.initializer)
            n = 0
            losses = 0
            accs = 0
            while True:
                try:
                    _, loss = sess.run([optimiser, cross_entropy])
                    losses += loss
                    sys.stdout.write("\rBatch %2d --- Latest Loss: %6.2f" % (n+1, loss))
                    sys.stdout.flush()
                    n += 1
                except tf.errors.OutOfRangeError:
                    break
            sess.run(iterator.initializer)
            m = 0
            while True:
                try:
                    acc = sess.run(accuracy)
                    accs += acc
                    m += 1
                except tf.errors.OutOfRangeError:
                    break
            losses = losses / n
            accs = accs / m
            print("Iter: {}, Loss: {:.4f}, Acc: {:.4f}".format(i, losses, accs))

def parse_func(data):
    record_defaults = [0.0]
    data = tf.decode_csv(data, record_defaults=record_defaults)
    data_ = tf.slice(data[0], [0], [91728])
    label = tf.slice(data[0], [91728], [3])
    data_ = tf.reshape(data_, [13, 84, 84, 1])
    
    return data_, label


if __name__ == "__main__":
    import sys
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    main()