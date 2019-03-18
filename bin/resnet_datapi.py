import os
import sys
import datetime

import pandas as pd
import numpy as np
import tensorflow as tf
from absl import app
from bin.util import *
from bin.modules import *

from bin.inception_v4 import inception_v4
from bin.all_conv_cnn import all_conv
from bin.inception_SE import inception_v4_se

from lib.config import SCREEN_RESOLUTION, MINIMAP_RESOLUTION, MAP_PATH, \
    REPLAYS_PARSED_DIR, REPLAY_DIR, REPO_DIR, STANDARD_VERSION



def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    versions = ['1_3d_10sup']  
    batch_size = 50
    cap = 7500
    epochs = 75
    a = 5
    b = 3
    split = int(cap*0.9)
    batches = int(split/batch_size)
    test_batches = int((cap-split)/batch_size)
    now = datetime.datetime.now()
    lr = 0.01

    tensorboard_dir = os.path.join(REPO_DIR, 'tensorboard_logs', 'Inceptionv4_SE', 'AdamOpt', 'LearningRate_'+str(lr)+"""  '_Repetitions_'+str(a)+'_'+str(b)+"""'_SampleSize_'+str(cap)+'_'+str(now.year)+'-'+str(now.month)+'-'+str(now.day)+'-'+str(now.hour)+'-'+str(now.minute))

    file = build_file_array(type='csv', version=versions)
    file, supp_diff = filter_close_matchups(file, 10, versions, 'csv')


    train_dataset = tf.data.TextLineDataset(file[:split-1], compression_type='GZIP')
    test_dataset = tf.data.TextLineDataset(file[split:cap], compression_type='GZIP')
    train_dataset = train_dataset.batch(13*84*84+3)
    test_dataset = test_dataset.batch(13*84*84+3)
    train_dataset = train_dataset.map(parse_func, num_parallel_calls=4)
    test_dataset = test_dataset.map(parse_func, num_parallel_calls=4)
    
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    
    train_dataset = train_dataset.prefetch(1)
    test_dataset = test_dataset.prefetch(1)

    iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    
    train_init_op = iter.make_initializer(train_dataset)
    test_init_op = iter.make_initializer(test_dataset)
        
    features, labels = iter.get_next()

    y_ = inception_v4_se(features)
    # y_ = all_conv(features)
    y = labels
    softmax = tf.nn.softmax(y_)

    #get trainable params
    para = get_params(tf.trainable_variables())
    print(para)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y))

    # add an optimiser
    optimiser = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(softmax, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # setup the initialisation operator
    init_op = tf.global_variables_initializer()
    # setup the save and restore functionality for variables 
    saver = tf.train.Saver()


    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

        sess.run(init_op)

        for i in range(epochs):
            sess.run(train_init_op)
            losses = 0
            accs = 0
            cross_accs = 0 
            n= 0
            while True:
                try:
                    _, loss = sess.run([optimiser, cross_entropy])
                    losses += loss / batches
                    sys.stdout.write("\rBatch %2d --- Latest Loss: %6.2f" % (n+1, loss))
                    sys.stdout.flush()
                    n += 1
                except tf.errors.OutOfRangeError:
                    break
            sess.run(train_init_op)
            while True:
                try:
                    acc = sess.run(accuracy)
                    accs += acc /batches
                except tf.errors.OutOfRangeError:
                    break
            sess.run(test_init_op)
            k = 0
            while True:
                try:
                    cross_acc = sess.run(accuracy)
                    cross_accs += cross_acc / test_batches
                except tf.errors.OutOfRangeError:
                    break
            print(" --- Iter: {}, Loss: {:10.2f}, Acc: {:.4f}, 2nd_Acc: {:.4f}, {}".format(i, losses, accs, cross_acc, n))
            train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=losses),
                                              tf.Summary.Value(tag='train_accuracy', simple_value=accs),
                                              tf.Summary.Value(tag='test_accuracy', simple_value=cross_accs)])

            summary_writer.add_summary(summary=train_summary, global_step=i)
            summary_writer.flush()
        
        print("\nTraining complete!")
        save_path = saver.save(sess, os.path.join(tensorboard_dir, "model.ckpt"))

def parse_func(data):
    record_defaults = [0.0]
    data = tf.decode_csv(data, record_defaults=record_defaults)
    data_ = tf.slice(data[0], [0], [91728])
    label = tf.slice(data[0], [91728], [3])
    x_ = tf.reshape(data_, [13, 84, 84, 1])

    # remove zermo layers
    x0 = tf.slice(x_, [0, 0, 0, 0], [1, 84, 84, 1])
    x1 = tf.slice(x_, [6, 0, 0, 0], [1, 84, 84, 1])
    x2 = tf.slice(x_, [8, 0, 0, 0], [1, 84, 84, 1])
    x3 = tf.slice(x_, [9, 0, 0, 0], [1, 84, 84, 1])
    x4 = tf.slice(x_, [10, 0, 0, 0], [1, 84, 84, 1])
    x5 = tf.slice(x_, [5, 0, 0, 0], [1, 84, 84, 1])
    x6 = tf.slice(x_, [2, 0, 0, 0], [1, 84, 84, 1])
    x7 = tf.slice(x_, [11, 0, 0, 0], [1, 84, 84, 1])
    prep_layers = tf.concat([x0, x1, x2, x3, x4, x5, x6, x7], 0)
    return prep_layers, label

def resnet(x_):
    a = 5
    b = 3
    num_classes = 3
    repetitions = [a,b]
    reg_factor = 1e-4
    block_fn = basic_block
    
    block_fn = get_block(block_fn)
    with tf.name_scope("First_Layer"):
        conv1 = conv_bn_relu(x_, filters=32, kernel_size=[1, 7, 7], strides=(1,2,2), kernel_regularizer=tf.keras.regularizers.l2(reg_factor), padding='VALID')
        param = print_layer_details(tf.contrib.framework.get_name_scope(), conv1.get_shape())
    # pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[3,3,3], strides=(2,2,2), padding='SAME')
    
    block = conv1
    #print(block.get_shape())
    filters = 32
    for i, r in enumerate(repetitions):
        with tf.name_scope("Residual_Block_"+str(i+1)):
            block = residual_block_3d(block, block_fn, filters=filters, repetitions=r, kernel_regularizer=tf.keras.regularizers.l2(reg_factor), is_first_layer=(i == 0), scope="Residual_Block_"+str(i+1))
            filters *= 2
        
    block_output = batch_norm(block)
    width = int(int(block.get_shape()[2])/2)
    height = int(int(block.get_shape()[3]))
    with tf.name_scope("Avg_Pooling"):
        pool2 = tf.layers.average_pooling3d(inputs=block_output, 
                                            pool_size=[1,
                                                       width,
                                                       height],
                                            strides=(1,width,height))
        param = print_layer_details(tf.contrib.framework.get_name_scope(), pool2.get_shape())            
    flatten1 = tf.layers.flatten(pool2)
    x_ = flatten1
    with tf.name_scope("Dense_Layer"):
        x_ = tf.layers.dense(inputs=x_, units=128, kernel_regularizer=tf.keras.regularizers.l2(reg_factor), activation=tf.nn.relu)
        param = print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
        
    y_ = tf.layers.dense(inputs=x_, units=num_classes, kernel_regularizer=tf.keras.regularizers.l2(reg_factor), activation=tf.nn.relu)

    return y_


if __name__ == "__main__":
    import sys
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    main()