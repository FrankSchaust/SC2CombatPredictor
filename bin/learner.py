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
from lib.config import REPLAYS_PARSED_DIR, REPLAY_DIR, REPO_DIR, STANDARD_VERSION



def main():
    ### define constants for training

    # supress tf-warning for RAM-usage
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    # declare all used sample versions 
    versions = ['1_3d_10sup', '1_3d', '1_3d_15sup']  
    # declare batching, how many samples are used for training and the number of training epochs
    batch_size = 50
    cap = 30000
    epochs = 50
    # declare if data augmentation is used
    data_augmentation = False
    # set train/test-split and number of batches
    split = int(cap*0.9)
    batches = int(split/batch_size)
    test_batches = int((cap-split)/batch_size)
    # set start time to calculate duration of training
    now = datetime.datetime.now()
    # set learning rate
    lr = 0.005

    # initialize directory in which the logs are going to be saved
    tensorboard_dir = os.path.join(REPO_DIR, 'tensorboard_logs', 'Inception_v4_noAug_noWeight', 'LearningRate_'+str(lr) +'_SampleSize_' + str(cap) + '_' +str(now.year)+'-'+str(now.month)+'-'+str(now.day)+'-'+str(now.hour)+'-'+str(now.minute))
    fil = [[], [], []]
    files = [[], [], []]
    supp = [[], [], []]
    # get files and filter them by supply for every specified version
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
    # f is the array of file-paths that are used to evaluate the accuracy by supply
    # s is the related supplies for each file in f
    f = files_[cap:]
    s = supp_[cap:]

    # initialize the datasets and declare that the input files are gzip-compressed
    train_dataset = tf.data.TextLineDataset(files_[:split-1], compression_type='GZIP')
    test_dataset = tf.data.TextLineDataset(files_[split:cap], compression_type='GZIP')
    # every sample consists of 91731 lines
    # every line is read individually by the TextLineDataset-Reader
    # therefore we need to batch them all together and parse them respectively
    train_dataset = train_dataset.batch(13*84*84+3)
    test_dataset = test_dataset.batch(13*84*84+3)
    # if data-augmentation is True we augment the data in parse_func
    if data_augmentation:
        train_dataset = train_dataset.map(parse_func, num_parallel_calls=16)
        test_dataset = test_dataset.map(parse_func, num_parallel_calls=16)
    # else we do not need augmentation
    else:
        train_dataset = train_dataset.map(parse_func_no_aug, num_parallel_calls=16)
        test_dataset = test_dataset.map(parse_func_no_aug, num_parallel_calls=16)
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    # the flat function is only needed if we add samples by augmentation
    if data_augmentation:
        train_dataset = train_dataset.map(flat_func, num_parallel_calls=16)    
        test_dataset = test_dataset.map(flat_func, num_parallel_calls=16)
    
    # to get accuracies for each supply difference 
    # we need to delare another dataset and parse it like the previous ones
    validation_dataset = tf.data.TextLineDataset(f, compression_type='GZIP')
    validation_dataset = validation_dataset.batch(13*84*84+3)

    validation_dataset = validation_dataset.map(parse_func_no_aug, num_parallel_calls=16)      
    validation_dataset = validation_dataset.batch(1)

    # prefetch(1) allows the datasets to load multiple batches in parallel
    train_dataset = train_dataset.prefetch(1)
    test_dataset = test_dataset.prefetch(1)
    validation_dataset = validation_dataset.prefetch(1)

    # create an iterator over the dataset and declare initialization methods
    iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iter.make_initializer(train_dataset)
    test_init_op = iter.make_initializer(test_dataset)
    val_init_op = iter.make_initializer(validation_dataset)
    
    # get features and labels from iterator
    features, labels = iter.get_next()

    # give the features to the networks
    # the one that is going to be evaluated is not in comments
    # returns the predictions

    # y_ = inception_v4_se(features)
    y_ = resnet(features)
    # y_ = all_conv(features)
    # y_ = inception_v4(features)
    
    # get the ground-truth labels 
    y = labels
    # apply softmax to the predictions
    softmax = tf.nn.softmax(y_)

    #get trainable params
    para = get_params(tf.trainable_variables())
    print(para)

    # loss without weights
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y))

    # helper to return labels and predictions to session
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
        # declare a last stop if you want to continue training an existing model
        last_stop = None
        # set a save path from with the trained model will be loaded
        save_path_ = os.path.join(tensorboard_dir)
        # if last_stop is declared try to import the meta graph and variable values from the saved model
        if not last_stop == None:
            saver = tf.train.import_meta_graph(os.path.join(save_path_, 'model_at_epoch-'+str(last_stop)+'.meta'))
            # if the specified checkpoint does not exist load the latest checkpoint
            if tf.train.checkpoint_exists(os.path.join(save_path_, 'model_at_epoch-'+str(last_stop))):
                saver.restore(sess, os.path.join(save_path_, 'model_at_epoch-'+str(last_stop)))
            else:
                saver.restore(sess, tf.train.latest_checkpoint(save_path_))
            # set the epoch range to an array of "last stop" to "end of training"
            epoch_range = np.arange(last_stop, epochs)
            tensorboard_dir = save_path_
        else:
            # if we start a new training the epoch range is 0 to "end of training"
            epoch_range = np.arange(0, epochs)
            # initialize all variables randomly
            sess.run(init_op)
        # initialize the FileWriter and save the session graph to the log data
        summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
        # initialize arrays to save predictions and corresponding labels, as well as starting time
        labels_by_epoch = []
        pred_by_epoch = []
        timestamp = datetime.datetime.now()   
        try:     
            for i in epoch_range:
                # get timestamp
                epoch_timestamp = datetime.datetime.now()        
                # initialize training dataset
                sess.run(train_init_op)
                # initialize logging variables
                losses = 0
                accs = 0
                taccs = 0
                n= 0
                pred = []
                pred_test = []
                gt = []
                gt_test = []
                # while there are samples to fetch in the dataset -> keep running
                while True:
                    try:
                        # run the optimizer function for the current batch
                        # and return cross entropy, resulting labels and predictions and accuracy
                        _, loss, res, acc= sess.run([optimiser, cross_entropy, results, accuracy])
                        # save gt and labels
                        gt = np.append(gt, res[0])
                        pred = np.append(pred, res[1])
                        # calculate loss and accuracy relative to number of batches
                        losses += loss / batches
                        accs += acc / batches
                        sys.stdout.write("\rBatch %2d of %2d" % (n+1, batches))
                        sys.stdout.flush()
                        n += 1
                    except tf.errors.OutOfRangeError:
                        break
                # initialize test dataset
                sess.run(test_init_op)
                # while there are samples to fetch in the dataset -> keep running
                while True:
                    try:
                        # run accuracy and return gt and labels
                        res_test, tacc = sess.run([results, accuracy])
                        # compute metrics
                        taccs += tacc/test_batches
                        gt_test = np.append(gt_test, res_test[0])
                        pred_test = np.append(pred_test, res_test[1])
                    except tf.errors.OutOfRangeError:
                        break
                # save labels and predictions
                labels_by_epoch = np.append(labels_by_epoch, [gt])
                pred_by_epoch = np.append(pred_by_epoch, [pred])
                # evaluate metrics precision, recall and f1-score for training results
                p, rc, f1, _ = precision_recall_fscore_support(gt, pred, average='weighted')

                # evaluate metrics precision, recall and f1-score for test results
                pt, rct, ft, _ = precision_recall_fscore_support(gt_test, pred_test, average='weighted')
                # calculate time spent for this epoch
                time_spent = datetime.datetime.now() - epoch_timestamp
                time_to_sec = time_spent.seconds
                print(" --- Iter: {:-2}, Loss: {:-15.2f} --- TRAINING --- Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}, Top-1-Error: {:.2f} --- TEST DATA --- Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}, Top-1-Error: {:.2f} --- Calculated in {} seconds".format(i+1, losses, p, rc, f1, 1-accs, pt, rct, ft, 1-taccs, time_spent.seconds))
                # write all metrics into log data for this step
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
                # save model for this step
                save_path = saver.save(sess, os.path.join(tensorboard_dir, "model_at_epoch"), global_step=i+1)
        except KeyboardInterrupt:
            df = pd.DataFrame(labels_by_epoch)
            os.makedirs(tensorboard_dir, exist_ok=True)
            df.to_csv(os.path.join(tensorboard_dir, 'labels.csv'))
            df = pd.DataFrame(pred_by_epoch)
            df.to_csv(os.path.join(tensorboard_dir, 'preds.csv'))                
        # print(labels_by_epoch)

        # save labels and predictions as csv data
        df = pd.DataFrame(labels_by_epoch)
        os.makedirs(tensorboard_dir, exist_ok=True)
        df.to_csv(os.path.join(tensorboard_dir, 'labels.csv'))
        df = pd.DataFrame(pred_by_epoch)
        df.to_csv(os.path.join(tensorboard_dir, 'preds.csv'))
        # calculate times used for training
        train_time = datetime.datetime.now() - timestamp
        print("\nTraining complete after {}!".format(train_time))

        # initialize empty list of unique entries to supply
        uniques = np.unique(s)
        t_sup_accs = {}
        c_sup_accs = {}
        for u in uniques:
            t_sup_accs[str(int(u*2))] = 0
            c_sup_accs[str(int(u*2))] = 0

        # initialize validation dataset
        sess.run(val_init_op)
        # run predicitons for every sample
        for su in s:
            cp = sess.run(correct_prediction)
            # save prediction depending on their supply difference
            if cp[0]:
                t_sup_accs[str(int(su*2))] += 1
            c_sup_accs[str(int(su*2))] += 1
        # evaluate all elements of the arrays to receive accuracies by supply and write those into log data
        for u in uniques:
            sup_acc = t_sup_accs[str(int(u*2))] / c_sup_accs[str(int(u*2))]
            sup_sum = tf.Summary(value=[tf.Summary.Value(tag='acc_by_supplydiff', simple_value=sup_acc)])

            summary_writer.add_summary(summary=sup_sum, global_step=int(u*2))
            summary_writer.flush()
            print('Acc for samples with supply difference of {}: {}'.format(u, sup_acc))

#################################################
### Helper Functions for the dataset pipeline ###
#################################################
# parse function without augmentation
# primarly used for the validation dataset
def parse_func_no_aug(data):
    record_defaults = [0.0]
    data = tf.decode_csv(data, record_defaults=record_defaults)
    data_ = tf.slice(data[0], [0], [91728])
    label = tf.slice(data[0], [91728], [3])
    x_ = tf.reshape(data_, [13, 84, 84, 1])
    x_ = tf.cast(x_, tf.float32)

    # remove zero layers
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

def parse_func(data, aug=False):
    record_defaults = [0.0]
    data = tf.decode_csv(data, record_defaults=record_defaults)
    data_ = tf.slice(data[0], [0], [91728])
    label = tf.slice(data[0], [91728], [3])
    x_ = tf.reshape(data_, [13, 84, 84, 1])
    x_ = tf.cast(x_, tf.float32)

    # remove zero layers
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
        label = tf.stack([label, label, label, label], axis = 0)
        prep_layers = tf.stack([prep[0], prep[1], prep[2], prep_layers], axis = 0)
    if not aug:
        x0_flipped = tf.image.rot90(x0, k=2)
        x1_flipped = tf.image.rot90(x1, k=2)
        x2_flipped = tf.image.rot90(x2, k=2)
        x3_flipped = tf.image.rot90(x3, k=2)
        x4_flipped = tf.image.rot90(x4, k=2)
        x5_flipped = tf.image.rot90(x5, k=2)
        x6_flipped = tf.image.rot90(x6, k=2)
        x7_flipped = tf.image.rot90(x7, k=2)
        prep_= tf.concat([x0_flipped, x1_flipped, x2_flipped, x3_flipped, x4_flipped, x5_flipped, x6_flipped, x7_flipped], 0)
        label_2 = tf.reverse(tf.roll(label, shift=1, axis=0), axis=[-1])
        prep_layers = tf.stack([prep_layers, prep_])
        label = tf.stack([label, label_2])
    return prep_layers, label

# function to flatten the samples and labels
def flat_func(features, labels):
    features = tf.reshape(features, [-1, 8, 84, 84, 1])
    labels = tf.reshape(labels, [-1, 3])
    
    return features, labels

if __name__ == "__main__":
    import sys
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    main()