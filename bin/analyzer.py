import os
import sys
import datetime

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, precision_recall_fscore_support
from absl import app
from bin.util import *
from bin.modules import *

from bin.inception_v4 import inception_v4
from bin.all_conv_cnn import all_conv
from bin.inception_se import inception_v4_se
from bin.resnet import resnet
from lib.config import MAP_PATH, \
    REPLAYS_PARSED_DIR, REPLAY_DIR, REPO_DIR, STANDARD_VERSION



def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    versions = ['1_3d_10sup', '1_3d', '1_3d_15sup' ]  
    cap = 90000
    scap = 105000

    # tensorboard_dir = os.path.join(REPO_DIR, 'tensorboard_logs', 'Inception_v4_noAug_noWeight', 'LearningRate_'+str(lr) +'_SampleSize_' + str(cap) + '_' +str(now.year)+'-'+str(now.month)+'-'+str(now.day)+'-'+str(now.hour)+'-'+str(now.minute))
    tensorboard_dir = os.path.join(REPO_DIR, 'tensorboard_logs', 'Inception_v4_SE', 'Inception_v4_SE_Shift_noWeight', 'LearningRate_0.005_SampleSize_30000_2019-5-18-12-59')
    fil = [[], [], []]
    files = [[], [], []]
    supp = [[], [], []]
    for n, v in enumerate(versions):
        fil[n] = build_file_array(type='csv', version=[v])
        file, su = filter_close_matchups(fil[n], 15, [v], 'csv')
        files[n] = file 
        supp[n] = su
    # get remaining files
    files_ = []
    supp_ = []
    for i in range(3):
        files_ = np.append(files_, files[i])
        supp_ = np.append(supp_, supp[i])

    f = files_[cap:scap]
    s = supp_[cap:scap]




    # get accuracies for each supply difference 
    validation_dataset = tf.data.TextLineDataset(f, compression_type='GZIP')
    validation_dataset = validation_dataset.batch(13*84*84+3)

    validation_dataset = validation_dataset.map(parse_func_no_aug, num_parallel_calls=16)      
    validation_dataset = validation_dataset.batch(1)

    iter = tf.data.Iterator.from_structure(validation_dataset.output_types, validation_dataset.output_shapes)

    val_init_op = iter.make_initializer(validation_dataset)
    
    features, labels = iter.get_next()

    
    # y_ = inception_v4_small(features)
    # y_ = testitest(features)
    y_ = inception_v4_se(features)
    # y_ = resnet(features)
    # y_ = all_conv(features)
    # y_ = mininet(features)
    # y_ = inception_v4(features)
    y = labels
    softmax = tf.nn.softmax(y_)

    #get trainable params
    para = get_params(tf.trainable_variables())
    print(para)

    # loss without weights
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y))

    results = tf.argmax(y,1), tf.argmax(softmax, 1)
    # add an optimiser
    
    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(softmax, 1))

    # setup the initialisation operator
    init_op = tf.global_variables_initializer()
    # setup the save and restore functionality for variables 
    saver = tf.train.Saver(max_to_keep=50)

    
    with tf.Session() as sess:
        
        last_stop = 20
        save_path_ = os.path.join(tensorboard_dir)
        if not last_stop == None:
            saver = tf.train.import_meta_graph(os.path.join(save_path_, 'model_at_epoch-'+str(last_stop)+'.meta'))
            if tf.train.checkpoint_exists(os.path.join(save_path_, 'model_at_epoch-'+str(last_stop))):
                saver.restore(sess, os.path.join(save_path_, 'model_at_epoch-'+str(last_stop)))
                print("Model from Epoch " + str(last_stop) + " restored..")
            else:
                saver.restore(sess, tf.train.latest_checkpoint(save_path_))
            tensorboard_dir = save_path_
        else:
            sess.run(init_op)
        os.makedirs(os.path.join(tensorboard_dir, 'Auswertung', 'Step'+str(last_stop)), exist_ok=True)
        summary_writer = tf.summary.FileWriter(os.path.join(tensorboard_dir, 'Auswertung', 'Step'+str(last_stop)), sess.graph)

        # initialize empty list of unique entries to supply
        uniques = np.unique(s)
        t_sup_accs = {}
        c_sup_accs = {}
        for u in uniques:
            t_sup_accs[str(int(u*2))] = 0
            c_sup_accs[str(int(u*2))] = 0

        label_arr = []
        pred_arr = []
        sess.run(val_init_op)
        for n, su in enumerate(s):
            cp, res, _ = sess.run([correct_prediction, results, y_])
            pred_arr = np.append(pred_arr, res[1])
            label_arr = np.append(label_arr,res[0])
            # print(pr)
            if cp[0]:
                t_sup_accs[str(int(su*2))] += 1
            c_sup_accs[str(int(su*2))] += 1
            sys.stdout.write("\rSample %2d from %2d evaluated.." % (n+1, scap-cap))
            sys.stdout.flush()


        for u in uniques:
            sup_acc = t_sup_accs[str(int(u*2))] / c_sup_accs[str(int(u*2))]
            sup_sum = tf.Summary(value=[tf.Summary.Value(tag='acc_by_supplydiff', simple_value=sup_acc)])

            summary_writer.add_summary(summary=sup_sum, global_step=int(u*2))
            summary_writer.flush()
            print('Acc for samples with supply difference of {}: {}'.format(u, sup_acc))
       
        label_ = [int(e) for e in label_arr]
        pred_ = [int(e) for e in pred_arr] 
        #print(label_)
        #print(pred_)
        print(classification_report(label_, pred_, ['0', '1', '2']))

### Important Metrics to consider
# Accuracy Train / Test
# Loss Train
# scikit metrics for triple

def parse_func_no_aug(data):
    record_defaults = [0.0]
    data = tf.decode_csv(data, record_defaults=record_defaults)
    data_ = tf.slice(data[0], [0], [91728])
    label = tf.slice(data[0], [91728], [3])
    x_ = tf.reshape(data_, [13, 84, 84, 1])
    x_ = tf.cast(x_, tf.float32)

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

if __name__ == "__main__":
    import sys
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    main()