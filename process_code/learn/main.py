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
import numpy as np
import tensorflow as tf
import argparse
import pickle


from config import TreeConfig, NoTreeConfig
from generate_utils import gothrough
from lstm_treecrf import LSTM_TREE_CRF
from read_utils import read_feature_vectors, create_vocabulary, \
    read_output_labels

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

            #if verbose and step % 30 == 0:
            #    print y_pred
            #    print y
            
            # Write content of y_pred to the predict_writer
            if predict_writer != None:
                pass
    
    '''
    Print performance for each slot
    '''
    if not is_training:
        if has_output:
            print_and_log("Number of correct predictions = %d, Percentage = %.3f" % 
                          (total_correct_pred, 1.0 * total_correct_pred/ (eval_iters * m.batch_size) ))
            
            print_and_log("Subject accuracy = %.5f" % (evals[0] / eval_iters))
            
            print_and_log("Verb accuracy = %.5f" % (evals[1] / eval_iters))
            
            print_and_log("Object accuracy = %.5f" % (evals[2] / eval_iters))
            
            print_and_log("Preposition accuracy = %.5f" % (evals[3] / eval_iters))
            
            print_and_log("Preposition dependent accuracy = %.5f" % (evals[4] / eval_iters))
        
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
                                help = "Where to save the model or to load the model. By default, the model is put in /logs/run_TIMESTAMP" )

    parser.add_argument('-d', '--dict_dir',  action='store',
                                help = "Where to save vocab dicts or to load them. By default, vocab dict files are put in /logs/run_TIMESTAMP" )
    
    parser.add_argument('-r', '--tree',  action='store_true',
                                help = "Whether to use the tree-Crf version. By default, just plain LSTM." )


    args = parser.parse_args()
    
    mode = args.mode
    
    log_dir = args.model_dir

    dict_dir = args.dict_dir

    use_tree = args.tree

    if use_tree:
        print 'USE TREE CRF'
    else:
        print "DON'T USE TREE CRF"
    
    if mode == TRAIN:
        current_time = datetime.datetime.now()
        time_str = '%s_%s_%s_%s_%s_%s' % (current_time.year, current_time.month, current_time.day, 
                              current_time.hour, current_time.minute, current_time.second)
    
        if not log_dir:
            log_dir = os.path.join('logs', 'run_' + time_str)
        
        if not os.path.isdir(log_dir):
            print('Train and output into directory ' + log_dir)
            os.makedirs(log_dir)
        else:
            print('Continue training model in directory ' + log_dir)
        
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
        vocab_dict = {} 
        
        if dict_dir == None:
            dict_dir = os.path.join(log_dir, 'dict')
        
        if not os.path.isdir(dict_dir):
            os.makedirs(dict_dir)
        
        for slot in ALL_SLOTS:
            dict_file = os.path.join( dict_dir, slot + '.dict')
            if os.path.isfile(dict_file):
                with open(dict_file, 'rb') as fh:
                    vocab_dict[slot] = pickle.load(fh)
            else:
                if vocab_dict == {}:
                    vocab_dict = create_vocabulary(TUPLE_TRAIN_FILE)

                print 'Save vocab_dict of ', slot, ' into ',  dict_file
                with open(dict_file, 'wb') as fh:
                    pickle.dump(vocab_dict[slot], fh)
    
    if mode in [VALIDATE, TEST, BLIND_TEST]:
        if log_dir:
            model_path = os.path.join(log_dir, "model.ckpt")
            
            if os.path.isfile(model_path):
                print('Test using model ' + model_path)
            else:
                raise Exception('Non existant model')
            
            if dict_dir == None:
                dict_dir = os.path.join(log_dir, 'dict')
            
            vocab_dict = {}
            for slot in ALL_SLOTS:
                dict_file = os.path.join( dict_dir, slot + '.dict')
                
                try:
                    with open(dict_file, 'rb') as fh:
                        vocab_dict[slot] = pickle.load(fh)

                    print 'Load vocab_dict of ', slot, ' from ',  dict_file
                except:
                    print 'Load vocab_dict of ', slot, ' has exception!'
        else:
            sys.exit("learning.py -t TEST -m model_path")
            
            
    train_dir = FEATURE_TRAIN_DIR
    test_dir = FEATURE_TEST_DIR
    val_dir = FEATURE_VAL_DIR
    blindtest_dir = FEATURE_BLINDTEST_DIR
    
    print 'Turn training feature vectors into batch form for LSTM training'
    
    if use_tree:
        config = TreeConfig(vocab_dict)
        eval_config = TreeConfig(vocab_dict)
        intermediate_config = TreeConfig(vocab_dict)
    else:
        config, eval_config, intermediate_config = [NoTreeConfig() for _ in xrange(3)]
    
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

    if mode == TRAIN:
        train_label_ids = read_output_labels(ORIGINAL_ANNOTATE_TRAIN_FILE, TUPLE_TRAIN_FILE, vocab_dict)
        val_label_ids = read_output_labels(ORIGINAL_ANNOTATE_VAL_FILE, TUPLE_VAL_FILE, vocab_dict)

    if mode == VALIDATE:
        val_label_ids = read_output_labels(ORIGINAL_ANNOTATE_VAL_FILE, TUPLE_VAL_FILE, vocab_dict)
        
    if mode == TEST:
        test_label_ids = read_output_labels(ORIGINAL_ANNOTATE_TEST_FILE, TUPLE_TEST_FILE, vocab_dict)
            
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        
        '''
        Model for training
        '''
        print('-------- Setup m model ---------')
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            try:
                config.tree.initiate_crf()
            except:
                pass
            m = LSTM_TREE_CRF(is_training=True, has_output = True, config=config)
        
        '''
        Model for evaluating over training data
        '''
        print('-------- Setup m_intermediate_test model ---------')
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            try:
                intermediate_config.tree.initiate_crf()
            except:
                pass
            m_intermediate_test = LSTM_TREE_CRF(is_training=False, has_output = True, config=intermediate_config)
        
        '''
        Model for evaluating over test data
        '''
        print('-------- Setup mtest model ----------')
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            try:
                eval_config.tree.initiate_crf()
            except:
                pass
            mtest = LSTM_TREE_CRF(is_training=False, has_output = True, config=eval_config)
        
        if mode == BLIND_TEST:
            '''
            Model for predicting over blind_test data
            '''
            print('-------- Setup mpredict model ----------')
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mpredict = LSTM_TREE_CRF(is_training=False, has_output = False, config=eval_config)
        
        # Create data generator
        def create_data_generator( mode ):
            if mode == TRAIN:
                feature_generator = read_feature_vectors ( train_dir )

            if mode == VALIDATE:
                feature_generator = read_feature_vectors ( val_dir )
                
            if mode == TEST:
                feature_generator = read_feature_vectors ( test_dir )
                
            if mode == BLIND_TEST:
                feature_generator = read_feature_vectors ( blindtest_dir )
            return feature_generator

        if mode == TRAIN:
            

            print_and_log('----------------TRAIN---------------')

            if os.path.isfile(model_path):
                m.saver.restore(session, model_path)
                print_and_log("Restore model saved in file: %s" % model_path)
            else:
                tf.global_variables_initializer().run()
             
            for i in range(config.max_max_epoch):
                
                print_and_log('-------------------------------')
                start_time = time.time()
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print_and_log("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

                train_feature_generator = create_data_generator(TRAIN)

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
                    train_feature_generator = create_data_generator(TRAIN)
                    test_perplexity = run_epoch(session, m_intermediate_test, 
                        train_feature_generator , 
                                                train_label_ids, m_intermediate_test.test_op, 
                                                is_training=False, verbose = False)
                    print_and_log('Run model on validate data')
                    val_feature_generator = create_data_generator(VALIDATE)
                    test_perplexity = run_epoch(session, m_intermediate_test, val_feature_generator, 
                                                val_label_ids, m_intermediate_test.test_op, 
                                                is_training=False, verbose = True)
                
                if i % config.save_epoch == 0:
                    start_time = time.time()
                    _model_path = m.saver.save(session, model_path)
                    print_and_log("Model saved in file: %s" % _model_path)
                    print_and_log("Time %.3f" % (time.time() - start_time) )
                #except ValueError:
                #    print_and_log("Value error, reload the most recent saved model")
                #    m.saver.restore(session, model_path)
                #    break
            
            _model_path = m.saver.save(session, model_path)
            print_and_log("Model saved in file: %s" % _model_path)
        
        if mode in [VALIDATE, TEST, BLIND_TEST]:
            m.saver.restore(session, model_path)
            print_and_log("Restore model saved in file: %s" % model_path)
        
        if mode == VALIDATE:
            # Run on the validating data
            print_and_log('--------------VALIDATE--------------')  
            print_and_log('Run model on validate data')
            val_feature_generator = create_data_generator(VALIDATE)
            test_perplexity = run_epoch(session, mtest, val_feature_generator, 
                                        val_label_ids, mtest.test_op, 
                                        is_training=False, verbose=True)
            
        if mode == TEST:
            print_and_log('-----------PUBLIC TEST--------------')  
            print_and_log('Run model on test data')
            test_feature_generator = create_data_generator(TEST)
            test_perplexity = run_epoch(session, mtest, test_feature_generator, 
                                        test_label_ids, mtest.test_op, 
                                        is_training=False, verbose=True)
        
        if mode == BLIND_TEST:
            print_and_log('-----------BLIND TEST--------------')  
            print_and_log('Run model on blind-test data')
            blindtest_feature_generator = create_data_generator(BLIND_TEST)
            run_epoch(session, mpredict, blindtest_feature_generator, 
                                        None, mpredict.test_op, 
                                        is_training=False, verbose=True, 
                                        has_output = False)