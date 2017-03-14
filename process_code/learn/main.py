'''
Created on Mar 3, 2017

@author: Tuan
'''
'''
Read feature vectors from a feature directory

Parameters:
----------
directory:  a directory store feature vectors

Return:
------
features: dictionary mapping from clip_file to feature vectors
'''

import datetime
import logging
import os
import shutil
import sys
import time

import argparse
from gensim.corpora.dictionary import Dictionary

from learn.config import TreeConfig
from learn.generate_utils import gothrough
from learn.lstm_treecrf import LSTM_TREE_CRF
from learn.read_utils import read_feature_vectors, create_vocabulary, \
    read_output_labels
import numpy as np
import tensorflow as tf
from utils import FEATURE_TRAIN_DIR, FEATURE_TEST_DIR, FEATURE_VAL_DIR, \
    FEATURE_BLINDTEST_DIR, TUPLE_TRAIN_FILE, ORIGINAL_ANNOTATE_TRAIN_FILE, \
    ORIGINAL_ANNOTATE_TEST_FILE, TUPLE_TEST_FILE, ORIGINAL_ANNOTATE_VAL_FILE, \
    TUPLE_VAL_FILE, TRAIN, TEST, BLIND_TEST, VALIDATE, ALL_SLOTS


def print_and_log(log_str):
    print (log_str)
    logging.info(log_str)
    
    
def run_epoch(session, m, data_generator, label_ids, eval_op, verbose=False, is_training=True, has_output = True, 
              summary_op = None, summary_writer = None, predict_writer = None):
    
    '''
    Runs the model on the given data.
    
    Parameters:
    -----------
    session:        Tensorflow session
    m:              Model to run on
    data_generator: Generate a batch of data, see  :meth:`read_utils.read_feature_vectors`
    label_ids:      Map from a clip file name to list of ids
    eval_op:        Operator for evaluation (train/test/predict)
    
    
    
    Return:
    -------
    average_cost:   If has_output, it is the average cost for each data sample
    '''
    
    costs = 0
    evals = np.zeros(len(m.dictionaries))
    cost_iters = 0
    eval_iters = 0
    state = [session.run(s) for s in m.initial_state]
    total_correct_pred = 0
    
    if verbose:
        print_and_log('------PRINT OUT PREDICTED AND CORRECT LABELS--------')
    
    for step, (x, y, z) in enumerate( gothrough( data_generator, label_ids, m.batch_size, m.num_steps) ):
        feed_dict = {}
        feed_dict[m.input_data] = x
        if has_output:
            feed_dict[m.targets] = y
        feed_dict[m.input_lengths] = z
        
        for i in xrange(len(m.initial_state)):
            feed_dict[m.initial_state[i]] = state[i]
        
        ops = []
        
        if summary_op:
            ops.append(summary_op)
            
        if has_output:
            ops.append(m.cost)
            
        ops += [ m.final_state, eval_op ]
        
        stuffs = session.run(ops, feed_dict)
        
        if summary_op:
            summary = stuffs[0]
            
            if summary_writer != None:
                summary_writer.add_summary(summary, step)
        
        if has_output:
            cost = stuffs[-3]
        
        state, eval_val = stuffs[-2:]
        
        '''
        Print cost after some step
        '''
        if has_output:
            costs += cost
            cost_iters += 1
        
            if verbose and step % 30 == 0:
                print_and_log("cost %.3f, costs %.3f, iters %d, Step %d, perplexity: %.3f" % 
                  (cost, costs, cost_iters, step, np.exp(costs / cost_iters)))
                
                
        if not is_training:
            if has_output:
                y_pred, eval_val = eval_val
                
                eval_iters += 1
                
                evals += eval_val
            
                correct_pred = np.sum(np.all([np.equal(y_pred[:, i], y[:,i]) \
                                    for i in xrange(len(m.dictionaries))], axis = 0))
                total_correct_pred += correct_pred
                
            else:
                y_pred = eval_val
            
            # Write content of y_pred to the predict_writer
            if predict_writer != None:
                pass
    
    '''
    Print performance for each slot
    '''
    if not is_training:
        if has_output:
            print_and_log("Number of correct predictions = %d, Percentage = %.3f" % 
                          (total_correct_pred, total_correct_pred/ (eval_iters * m.batch_size) ))
            
            print_and_log("Subject accuracy = %.5f" % (evals[0] / eval_iters))
            
            print_and_log("Object accuracy = %.5f" % (evals[1] / eval_iters))
            
            print_and_log("Theme accuracy = %.5f" % (evals[2] / eval_iters))
            
            print_and_log("Event accuracy = %.5f" % (evals[3] / eval_iters))
            
            print_and_log("Preposition accuracy = %.5f" % (evals[4] / eval_iters))
        
    if has_output:    
        return np.exp(costs / cost_iters)

if __name__ == '__main__':
    # ========================================================================
    # ========================================================================
    # ===========================SETUP TRAIN TEST=============================
    parser = argparse.ArgumentParser(description='A script to train and test movie dataset using LSTM-CRF')
    
    parser.add_argument('-t', '--mode', action = 'store', 
                                help = "Store mode. Pick between TRAIN, VALIDATE, TEST and BLIND_TEST. TRAIN actually means TRAIN + VALIDATE" )
    
    parser.add_argument('-m', '--model_dir',  action='store',
                                help = "Where to save the model or to load the model. By default, the model and its auxiliary files are put in /logs/run_TIMESTAMP" )
    
    args = parser.parse_args()
    
    mode = args.mode
    
    log_dir = args.model_dir
    
    if mode == TRAIN:
        current_time = datetime.datetime.now()
        time_str = '%s_%s_%s_%s_%s_%s' % (current_time.year, current_time.month, current_time.day, 
                              current_time.hour, current_time.minute, current_time.second)
    
        if not log_dir:
            log_dir = os.path.join('logs', 'run_' + time_str)
            
        print('Train and output into directory ' + log_dir)
        os.makedirs(log_dir)
        logging.basicConfig(filename = os.path.join(log_dir, 'logs.log'),level=logging.DEBUG)
        
        # Copy the current executed py file to log (To make sure we can replicate the experiment with the same code)
        shutil.copy(os.path.realpath(__file__), log_dir)
        
        '''
        Create model path
        '''
        model_path = os.path.join(log_dir, "model.ckpt")
        
        '''
        Create gensim dictionary from the tokens in TUPLE_TRAIN_FILE
        '''
        vocab_dict = create_vocabulary(TUPLE_TRAIN_FILE)
        
        dict_dir = os.path.join(log_dir, 'dict')
        
        os.makedirs(dict_dir)
        
        for slot in vocab_dict:
            dict_file = os.path.join( dict_dir, slot + '.dict')
            vocab_dict[slot].save(dict_file)
    
    if mode in [VALIDATE, TEST, BLIND_TEST]:
        if log_dir:
            model_path = os.path.join(log_dir, "model.ckpt")
            
            print('Test using model ' + model_path)
            
            dict_dir = os.path.join(log_dir, 'dict')
            
            vocab_dict = {}
            for slot in ALL_SLOTS:
                dict_file = os.path.join( dict_dir, slot + '.dict')
                
                vocab_dict[slot] = Dictionary()
                vocab_dict[slot].load( dict_file )
        else:
            sys.exit("learning.py -t TEST -m model_path")
            
            
    train_dir = FEATURE_TRAIN_DIR
    test_dir = FEATURE_TEST_DIR
    val_dir = FEATURE_VAL_DIR
    blindtest_dir = FEATURE_BLINDTEST_DIR
    
    
    if mode == TRAIN:
        train_feature_generator = read_feature_vectors ( train_dir )
        
        train_label_ids = read_output_labels(ORIGINAL_ANNOTATE_TRAIN_FILE, TUPLE_TRAIN_FILE, vocab_dict)
        
        val_feature_generator = read_feature_vectors ( val_dir )
        
        val_label_ids = read_output_labels(ORIGINAL_ANNOTATE_VAL_FILE, TUPLE_VAL_FILE, vocab_dict)
    
    if mode == VALIDATE:
        val_feature_generator = read_feature_vectors ( val_dir )
        
        val_label_ids = read_output_labels(ORIGINAL_ANNOTATE_VAL_FILE, TUPLE_VAL_FILE, vocab_dict)
        
    if mode == TEST:
        test_feature_generator = read_feature_vectors ( test_dir )
        
        test_label_ids = read_output_labels(ORIGINAL_ANNOTATE_TEST_FILE, TUPLE_TEST_FILE, vocab_dict)
        
    if mode == BLIND_TEST:
        blindtest_feature_generator = read_feature_vectors ( blindtest_dir )
    
    print 'Turn training feature vectors into batch form for LSTM training'
    
    config = TreeConfig(vocab_dict)
    eval_config = TreeConfig(vocab_dict)
    intermediate_config = TreeConfig(vocab_dict)
    
    '''
    No dropout
    '''
    intermediate_config.keep_prob = 1
    '''
    No droupout
    Single input
    Decrease CRF weight 
    '''
    eval_config.keep_prob = 1
    eval_config.batch_size = 20
    eval_config.crf_weight = 0.5
    
    logging.info("Train Configuration")
    for attr in dir(config):
        # Not default properties
        if attr[:2] != '__':
            log_str = "%s = %s" % (attr, getattr(config, attr))
            logging.info(log_str)

    logging.info("Evaluation Configuration")
    for attr in dir(eval_config):
        # Not default properties
        if attr[:2] != '__':
            log_str = "%s = %s" % (attr, getattr(eval_config, attr))
            logging.info(log_str)
            
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        
        '''
        Model for training
        '''
        print('-------- Setup m model ---------')
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            config.tree.initiate_crf()
            m = LSTM_TREE_CRF(is_training=True, config=config)
        
        '''
        Model for evaluating over training data
        '''
        print('-------- Setup m_intermediate_test model ---------')
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            intermediate_config.tree.initiate_crf()
            m_intermediate_test = LSTM_TREE_CRF(is_training=False, config=intermediate_config)
        
        '''
        Model for evaluating over test data
        '''
        print('-------- Setup mtest model ----------')
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            eval_config.tree.initiate_crf()
            mtest = LSTM_TREE_CRF(is_training=False, config=eval_config)
        
        if mode == BLIND_TEST:
            '''
            Model for predicting over blind_test data
            '''
            print('-------- Setup mpredict model ----------')
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mpredict = LSTM_TREE_CRF(is_training=False, has_output = False, config=eval_config)
            
        if mode == TRAIN:
            tf.global_variables_initializer().run()

            print_and_log('----------------TRAIN---------------')
             
            for i in range(config.max_max_epoch):
                try:
                    print_and_log('-------------------------------')
                    start_time = time.time()
                    lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                    m.assign_lr(session, config.learning_rate * lr_decay)

                    print_and_log("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

                    train_perplexity = run_epoch(session, m, train_feature_generator, 
                                                 train_label_ids, m.train_op,
                                               verbose=True)
                    print_and_log("Epoch: %d Train Perplexity: %s" % (i + 1, str(train_perplexity)))
                    print_and_log("Time %.3f" % (time.time() - start_time) )
                    print_and_log('-------------------------------') 

                    if i % config.test_epoch == 0:
                        print_and_log('----------Intermediate test -----------')  
                        # Run test on train
                        print_and_log('Run model on train data')
                        test_perplexity = run_epoch(session, m_intermediate_test, train_feature_generator, 
                                                    train_label_ids, m_intermediate_test.test_op, 
                                                    is_training=False, verbose = False)
                        print_and_log('Run model on validate data')
                        test_perplexity = run_epoch(session, m_intermediate_test, val_feature_generator, 
                                                    val_label_ids, m_intermediate_test.test_op, 
                                                    is_training=False, verbose = True)
                    
                    if i % config.save_epoch == 0:
                        start_time = time.time()
                        _model_path = m.saver.save(session, model_path)
                        print_and_log("Model saved in file: %s" % _model_path)
                        print_and_log("Time %.3f" % (time.time() - start_time) )
                except ValueError:
                    print_and_log("Value error, reload the most recent saved model")
                    m.saver.restore(session, model_path)
                    break
            
            _model_path = m.saver.save(session, model_path)
            print_and_log("Model saved in file: %s" % _model_path)
        
        if mode in [VALIDATE, TEST, BLIND_TEST]:
            m.saver.restore(session, model_path)
            print_and_log("Restore model saved in file: %s" % model_path)
        
        if mode == VALIDATE:
            # Run on the validating data
            print_and_log('--------------VALIDATE--------------')  
            print_and_log('Run model on validate data')
            test_perplexity = run_epoch(session, mtest, val_feature_generator, 
                                        val_label_ids, mtest.test_op, 
                                        is_training=False, verbose=True)
            
        if mode == TEST:
            print_and_log('-----------PUBLIC TEST--------------')  
            print_and_log('Run model on test data')
            test_perplexity = run_epoch(session, mtest, test_feature_generator, 
                                        test_label_ids, mtest.test_op, 
                                        is_training=False, verbose=True)
        
        if mode == BLIND_TEST:
            print_and_log('-----------BLIND TEST--------------')  
            print_and_log('Run model on blind-test data')
            run_epoch(session, mpredict, blindtest_feature_generator, 
                                        None, mpredict.test_op, 
                                        is_training=False, verbose=True, 
                                        has_output = False)