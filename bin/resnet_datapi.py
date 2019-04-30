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
from bin.inception_SE import inception_v4_se
from bin.inception_v4_small import inception_v4_small
from lib.config import SCREEN_RESOLUTION, MINIMAP_RESOLUTION, MAP_PATH, \
    REPLAYS_PARSED_DIR, REPLAY_DIR, REPO_DIR, STANDARD_VERSION



def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    versions = ['1_3d_10sup', '1_3d', '1_3d_15sup']  
    batch_size = 20
    cap = 90000
    epochs = 50
    data_augmentation = True
    a = 5
    b = 3
    split = int(cap*0.9)
    batches = int(split/batch_size)
    test_batches = int((cap-split)/batch_size)
    now = datetime.datetime.now()
    lr = 0.001

    tensorboard_dir = os.path.join(REPO_DIR, 'tensorboard_logs', 'Inception_v4_SE', 'LearningRate_'+str(lr)+'_SampleSize_'+str(cap)+'_'+str(now.year)+'-'+str(now.month)+'-'+str(now.day)+'-'+str(now.hour)+'-'+str(now.minute))
    fil = [[], [], []]
    files = [[], [], []]
    supp = [[], [], []]
    for n, v in enumerate(versions):
        fil[n] = build_file_array(type='csv', version=[v])
        file, su = filter_close_matchups(fil[n], 10, [v], 'csv')
        files[n] = file 
        supp[n] = su
    # get remaining files
    files_ = []
    supp_ = []
    for i in range(3):
        files_ = np.append(files_, files[i])
        supp_ = np.append(supp_, supp[i])

    f = files_[cap:]
    s = supp_[cap:]


    train_dataset = tf.data.TextLineDataset(files_[:split-1], compression_type='GZIP')
    test_dataset = tf.data.TextLineDataset(files_[split:cap], compression_type='GZIP')
    train_dataset = train_dataset.batch(13*84*84+3)
    test_dataset = test_dataset.batch(13*84*84+3)
    train_dataset = train_dataset.map(parse_func, num_parallel_calls=16)
    test_dataset = test_dataset.map(parse_func, num_parallel_calls=16)
    
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    if data_augmentation:
        train_dataset = train_dataset.map(flat_func, num_parallel_calls=16)    
        test_dataset = test_dataset.map(flat_func, num_parallel_calls=16)

    # get accuracies for each supply difference 
    validation_dataset = tf.data.TextLineDataset(f, compression_type='GZIP')
    validation_dataset = validation_dataset.batch(13*84*84+3)

    validation_dataset = validation_dataset.map(parse_func_no_aug, num_parallel_calls=16)      
    validation_dataset = validation_dataset.batch(1)


    train_dataset = train_dataset.prefetch(1)
    test_dataset = test_dataset.prefetch(1)
    iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iter.make_initializer(train_dataset)
    test_init_op = iter.make_initializer(test_dataset)
    val_init_op = iter.make_initializer(validation_dataset)
    
    features, labels = iter.get_next()

    
    y_ = inception_v4_small(features)
    # y_ = testitest(features)
    # y_ = inception_v4_se(features)
    # y_ = resnet(features)
    # y_ = all_conv(features)
    # y_ = inception_v4(features)
    y = labels
    softmax = tf.nn.softmax(y_)

    #get trainable params
    para = get_params(tf.trainable_variables())
    print(para)

    class_weights = tf.constant([[1.0, 5.0, 9.0]])
    label_weights = tf.reduce_sum(class_weights * y, axis=1)
    cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=y_, onehot_labels=y, weights=label_weights))

    results = tf.argmax(y,1), tf.argmax(softmax, 1)
    # add an optimiser
    optimiser = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(softmax, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # setup the initialisation operator
    init_op = tf.global_variables_initializer()
    # setup the save and restore functionality for variables 
    saver = tf.train.Saver(max_to_keep=50)

    
    with tf.Session() as sess:
        
        last_stop = None
        save_path_ = os.path.join(tensorboard_dir)
        if not last_stop == None:
            saver = tf.train.import_meta_graph(os.path.join(save_path_, 'model_at_epoch-'+str(last_stop)+'.meta'))
            if tf.train.checkpoint_exists(os.path.join(save_path_, 'model_at_epoch-'+str(last_stop))):
                saver.restore(sess, os.path.join(save_path_, 'model_at_epoch-'+str(last_stop)))
            else:
                saver.restore(sess, tf.train.latest_checkpoint(save_path_))
            epoch_range = np.arange(last_stop, epochs)
            tensorboard_dir = save_path_
        else:
            epoch_range = np.arange(0, epochs)
            sess.run(init_op)
        summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
        labels_by_epoch = []
        pred_by_epoch = []
        timestamp = datetime.datetime.now()        
        for i in epoch_range:
            #get timestamp
            epoch_timestamp = datetime.datetime.now()        
            sess.run(train_init_op)
            losses = 0
            accs = 0
            taccs = 0
            n= 0
            pred = []
            pred_test = []
            gt = []
            gt_test = []
            while True:
                try:
                    _, loss, res, acc = sess.run([optimiser, cross_entropy, results, accuracy])
                    gt = np.append(gt, res[0])
                    pred = np.append(pred, res[1])
                    losses += loss / batches
                    accs += acc / batches
                    sys.stdout.write("\rBatch %2d of %2d" % (n+1, batches))
                    sys.stdout.flush()
                    n += 1
                except tf.errors.OutOfRangeError:
                    break
            sess.run(test_init_op)
            while True:
                try:
                    res_test, tacc = sess.run([results, accuracy])
                    taccs += tacc/test_batches
                    gt_test = np.append(gt_test, res_test[0])
                    pred_test = np.append(pred_test, res_test[1])
                except tf.errors.OutOfRangeError:
                    break
            labels_by_epoch = np.append(labels_by_epoch, [gt])
            pred_by_epoch = np.append(pred_by_epoch, [pred])
            p, rc, f1, _ = precision_recall_fscore_support(gt, pred, average='weighted')

            # print(gt, pred)
            pt, rct, ft, _ = precision_recall_fscore_support(gt_test, pred_test, average='weighted')
            time_spent = datetime.datetime.now() - epoch_timestamp
            time_to_sec = time_spent.seconds
            print(" --- Iter: {:-2}, Loss: {:-15.2f} --- TRAINING --- Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}, Top-1-Error: {:.2f} --- TEST DATA --- Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}, Top-1-Error: {:.2f} --- Calculated in {} seconds".format(i+1, losses, p, rc, f1, 1-accs, pt, rct, ft, 1-taccs, time_spent.seconds))
            # print(classification_report(gt, pred, target_names=['Minerals', 'Vespene', 'Remis']))
            train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=losses),
                                              tf.Summary.Value(tag='train_precision', simple_value=p),
                                              tf.Summary.Value(tag='test_precision', simple_value=pt),
                                              tf.Summary.Value(tag='train_recall', simple_value=rc),
                                              tf.Summary.Value(tag='test_recall', simple_value=rct),
                                              tf.Summary.Value(tag='train_f1_score', simple_value=f1),
                                              tf.Summary.Value(tag='test_f1_score', simple_value=ft),
                                              tf.Summary.Value(tag='train_accuracy', simple_value=accs),
                                              tf.Summary.Value(tag='test_accuracy', simple_value=taccs),
                                              tf.Summary.Value(tag='time_spent_seconds', simple_value=time_to_sec)])

            summary_writer.add_summary(summary=train_summary, global_step=i)
            summary_writer.flush()
            save_path = saver.save(sess, os.path.join(tensorboard_dir, "model_at_epoch"), global_step=i+1)
        # print(labels_by_epoch)
        df = pd.DataFrame(labels_by_epoch)
        os.makedirs(tensorboard_dir, exist_ok=True)
        df.to_csv(os.path.join(tensorboard_dir, 'labels.csv'))
        df = pd.DataFrame(pred_by_epoch)
        df.to_csv(os.path.join(tensorboard_dir, 'preds.csv'))
        train_time = datetime.datetime.now() - timestamp
        print("\nTraining complete after {}!".format(train_time))
        training = False
        # initialize empty list of unique entries to supply
        uniques = np.unique(s)
        t_sup_accs = {}
        c_sup_accs = {}
        for u in uniques:
            t_sup_accs[str(int(u*2))] = 0
            c_sup_accs[str(int(u*2))] = 0


        sess.run(val_init_op)
        for su in s:
            cp = sess.run(correct_prediction)
            if cp[0]:
                t_sup_accs[str(int(su*2))] += 1
            c_sup_accs[str(int(su*2))] += 1

        for u in uniques:
            sup_acc = t_sup_accs[str(int(u*2))] / c_sup_accs[str(int(u*2))]
            sup_sum = tf.Summary(value=[tf.Summary.Value(tag='acc_by_supplydiff', simple_value=sup_acc)])

            summary_writer.add_summary(summary=sup_sum, global_step=int(u*2))
            summary_writer.flush()
            print('Acc for samples with supply difference of {}: {}'.format(u, sup_acc))

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

def parse_func(data, aug=True):
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
    if aug:
        prep = [[], [], []]
        for k in range(3):
            x0_flipped = tf.image.rot90(x0, k=k+1)
            x1_flipped = tf.image.rot90(x1, k=k+1)
            x2_flipped = tf.image.rot90(x2, k=k+1)
            x3_flipped = tf.image.rot90(x3, k=k+1)
            x4_flipped = tf.image.rot90(x4, k=k+1)
            x5_flipped = tf.image.rot90(x5, k=k+1)
            x6_flipped = tf.image.rot90(x6, k=k+1)
            x7_flipped = tf.image.rot90(x7, k=k+1)
            prep[k]= tf.concat([x0_flipped, x1_flipped, x2_flipped, x3_flipped, x4_flipped, x5_flipped, x6_flipped, x7_flipped], 0)
        prep_layers = tf.stack([prep[0], prep[1], prep[2], prep_layers], axis = 0)
        label = tf.stack([label, label, label, label], axis = 0)
    return prep_layers, label

def flat_func(features, labels):
    features = tf.reshape(features, [-1, 8, 84, 84, 1])
    labels = tf.reshape(labels, [-1, 3])
    
    return features, labels
        

def resnet(x_):
    a = 5
    b = 3
    num_classes = 3
    repetitions = [a,b]
    reg_factor = 1e-4
    block_fn = basic_block
    
    block_fn = get_block(block_fn)
    with tf.name_scope("First_Layer"):
        conv1 = conv_bn_relu(x_, filters=64, kernel_size=[1, 7, 7], strides=(1,2,2), kernel_regularizer=tf.keras.regularizers.l2(reg_factor), padding='VALID')
        print_layer_details(tf.contrib.framework.get_name_scope(), conv1.get_shape())
        pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[1,3,3], strides=(1,2,2), padding='SAME')
    
    block = pool1
    #print(block.get_shape())
    filters = 64
    for i, r in enumerate(repetitions):
        with tf.name_scope("Residual_Block_"+str(i+1)):
            block = residual_block_3d(block, block_fn, filters=filters, repetitions=r, kernel_regularizer=tf.keras.regularizers.l2(reg_factor), is_first_layer=(i == 0), scope="Residual_Block_"+str(i+1))
            filters *= 2
        
    block_output = batch_norm(block)
    # width = int(int(block.get_shape()[2])/2)
    # height = int(int(block.get_shape()[3])/2)
    # with tf.name_scope("Avg_Pooling"):
    #     pool2 = tf.layers.average_pooling3d(inputs=block_output, 
    #                                        pool_size=[1,
    #                                                   width,
    #                                                   height],
    #                                        strides=(1,width,height))
    #    print_layer_details(tf.contrib.framework.get_name_scope(), pool2.get_shape())            
    flatten1 = tf.layers.flatten(block_output)
    x_ = flatten1
    # with tf.name_scope("Dense_Layer"):
    #     x_ = tf.layers.dense(inputs=x_, units=64, kernel_regularizer=tf.keras.regularizers.l2(reg_factor), activation=tf.nn.relu)
    #     print_layer_details(tf.contrib.framework.get_name_scope(), x_.get_shape())
        
    y_ = tf.layers.dense(inputs=x_, units=num_classes)

    return y_

def testitest(x_):
    with tf.name_scope("First_Auxilliary"):
        height = int(x_.get_shape()[2])
        x_avg = tf.layers.average_pooling3d(inputs=x_, pool_size=[1,height,height], strides=(1,height,height))
        x_flat = tf.layers.flatten(x_avg)
        y_1 = tf.layers.dense(x_flat, units=3)
    
    with tf.name_scope("Conv_Max"):
        x_conv = tf.layers.conv3d(inputs=x_, filters=32, kernel_size=[1, 3, 3], strides=(1,1,1), padding='same', activation=tf.nn.relu)
        x_max = tf.layers.max_pooling3d(inputs=x_conv, pool_size=[1,3,3], strides=(1,2,2))
        print_layer_details(tf.contrib.framework.get_name_scope(), x_max.get_shape())
    with tf.name_scope("Conv_Strides"):
        x_conv_s = tf.layers.conv3d(inputs=x_, filters=32, kernel_size=[1, 3, 3], strides=(1,2,2), padding='Valid', activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_conv_s.get_shape())
    x_add = tf.keras.layers.add([x_max, x_conv_s])
    x_add = tf.layers.batch_normalization(inputs=x_add, training=True)

    with tf.name_scope("Second_Auxilliary"):
        height = int(x_add.get_shape()[2])
        x_avg = tf.layers.average_pooling3d(inputs=x_add, pool_size=[1,height,height], strides=(1,height,height))
        x_flat = tf.layers.flatten(x_avg)
        y_2 = tf.layers.dense(x_flat, units=3)
    with tf.name_scope("Full_Conv"):
        depth = int(x_add.get_shape()[1])
        height = int(x_add.get_shape()[2])
        x_full = tf.layers.conv3d(inputs=x_add, filters=2, kernel_size=[depth, height, height], strides=(1,1,1), padding='Valid', activation=tf.nn.relu)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_full.get_shape())
    with tf.name_scope("Dim_Red"):
        x_dr = tf.layers.conv3d(inputs=x_add, filters=2, kernel_size=[1, 3, 3], strides=(1,2,2), padding='Valid', activation=tf.nn.relu)
        x_mul = x_dr * x_full
        x_mul = tf.layers.batch_normalization(inputs=x_mul, training=True)
        print_layer_details(tf.contrib.framework.get_name_scope(), x_mul.get_shape())
    
    with tf.name_scope("Third_Auxilliary"):
        height = int(x_mul.get_shape()[2])
        x_avg = tf.layers.average_pooling3d(inputs=x_mul, pool_size=[1,height,height], strides=(1,height,height))
        x_flat = tf.layers.flatten(x_avg)
        y_3 = tf.layers.dense(x_flat, units=3)
    for i in range(6):
        with tf.name_scope('Conv_Pipe_'+str(i)):
            x_sym = tf.layers.conv3d(inputs=x_mul, filters=2*(i+3), kernel_size=[1, 3, 3], strides=(1,1,1), padding='Valid', activation=tf.nn.relu)

            x_edg = tf.layers.conv3d(inputs=x_mul, filters=2*(i+3), kernel_size=[1, 1, 3], strides=(1,1,1), padding='Valid', activation=tf.nn.relu)
            x_edg = tf.layers.conv3d(inputs=x_edg, filters=2*(i+3), kernel_size=[1, 3, 1], strides=(1,1,1), padding='Valid', activation=tf.nn.relu)
            x_mul = tf.concat([x_sym, x_edg], 4)
            x_mul = tf.layers.batch_normalization(inputs=x_mul, training=True)
            print_layer_details(tf.contrib.framework.get_name_scope(), x_mul.get_shape())
    x_ = tf.layers.flatten(x_mul)
    y_4 = tf.layers.dense(x_, units=3)
    with tf.name_scope("Final_Mult"):
        y_comb = y_1*y_2*y_3*y_4
        print_layer_details(tf.contrib.framework.get_name_scope(), y_comb.get_shape())
    return y_comb

if __name__ == "__main__":
    import sys
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    main()