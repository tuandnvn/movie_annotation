'''
Created on Mar 6, 2017

@author: Tuan
'''
from collections import deque
from copy import deepcopy
import copy

import numpy as np
import tensorflow as tf

from crf_utils import gather_2d, gather_2d_to_shape, expand, expand_first

try:
    from tensorflow.nn.rnn_cell import BasicLSTMCell, DropoutWrapper, MultiRNNCell
except:
    from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, MultiRNNCell

class LSTM_TREE_CRF(object):
    '''
    '''


    def __init__(self, is_training, has_output, config):
        '''
        Parameters:
        ----------
        is_training: Whether we're training over the data or not (whether we should update the model parameters)
        has_output:  Whether there are target to calculate loss or we just output predicted labels
        config shoule have:
            config.tree = Tree
            
        '''
        try:
            self.tree = config.tree
        except:
            self.tree = None

        self.batch_size = batch_size = config.batch_size
        # Maximum number of steps in each data sequence
        self.num_steps = num_steps = config.num_steps
        self.n_input = n_input = config.n_input
        self.max_grad_norm = config.max_grad_norm
        self.size = size = config.hidden_size

        self.crf_weight = crf_weight = config.crf_weight
        self.node_types = config.node_types
        
        self.hidden_layer = hidden_layer = config.hidden_layer

        # self.null_weights = config.null_weights

        self.loss_weights = config.loss_weights
        self.balance = config.balance

        print 'Use hidden_layer = ', hidden_layer

        # self.dictionaries is dict of dict
        self.dictionaries = config.dictionaries

        self.n_labels = len(self.node_types)

        self.train_algo = config.train_algo

        #self.null_discounts = dict([(slot, 
        #                            tf.one_hot(self.dictionaries[slot].token2id['null'], 
        #                                        len(self.dictionaries[slot]), 
        #                                        on_value=self.null_weights, 
        #                                        off_value=1.0, dtype = tf.float32 )) 
        #                        for slot in self.node_types])

        
        
        # Input data and labels should be set as placeholders
        self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps, n_input])
        
        if has_output:
            self._targets = tf.placeholder(tf.int32, [batch_size, self.n_labels])
        
        # Length for self._input_data
        self._input_lengths = tf.placeholder(tf.int32, [batch_size] )

        self._debug = []
        
        # self.n_labels cells for self.n_labels outputs
        #if hidden_layer:
        #    cell_size = size
        #else:
        #    cell_size = n_input
        lstm_cells = [BasicLSTMCell(size, forget_bias = 0.0, state_is_tuple=True)\
                      for _ in xrange(self.n_labels)]


        # DropoutWrapper is a decorator that adds Dropout functionality
        if is_training and config.keep_prob < 1:
            lstm_cells = [DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)\
                              for lstm_cell in lstm_cells]
        cells = [MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)\
                 for lstm_cell in lstm_cells]
        
        # Initial states of the cells
        # cell.state_size = config.num_layers * 2 * size
        # Size = self.n_labels x ( batch_size x cell.state_size )
        self._initial_state = [cell.zero_state(batch_size, tf.float32) for cell in cells]
        
        
        if hidden_layer:
            # Transformation of input to a list of num_steps data points
            # For tf.nn.rnn
            # inputs = tf.transpose(self._input_data, [1, 0, 2]) #(num_steps, batch_size, n_input)
            inputs = tf.reshape(self._input_data, [-1, n_input]) # (batch_size * num_steps, n_input)

            with tf.variable_scope("hidden"):
                weight = tf.get_variable("weight", [n_input, size])
                bias = tf.get_variable("bias", [size])
                
                # (batch_size * num_steps, size)
                inputs = tf.matmul(inputs, weight) + bias
            inputs = tf.reshape(inputs, (-1, num_steps, size)) # (batch_size, num_steps, size)
        else:
            inputs = self._input_data

        # For tf.nn.rnn
        # inputs = tf.split(0, num_steps, inputs) # num_steps * ( batch_size, size )
        
        outputs_and_states = []
        
        # A list of n_labels values
        # Each value is (output, state)
        # output is of size:  ( batch_size, num_steps, size )
        # state is of size:   ( batch_size, cell.state_size )
        for i in xrange(self.n_labels):
            with tf.variable_scope("lstm" + str(i)):
                # Old code, use tf.nn.rnn
                # output_and_state = tf.nn.rnn(cells[i], inputs, initial_state = self._initial_state[i])
                
                # New code, use tf.nn.dynamic_rnn
                output_and_state = tf.nn.dynamic_rnn(cells[i], inputs, dtype=tf.float32, initial_state = self._initial_state[i], 
                                                     sequence_length = self._input_lengths)
                outputs_and_states.append(output_and_state)
                
        
        
        # n_labels x ( batch_size, size )
        # For tf.nn.rnn
        # outputs = [output_and_state[0][-1]\
        #           for output_and_state in outputs_and_states]
        
        # n_labels x ( num_steps, batch_size, size )
        outputs = [tf.transpose(output_and_state[0], [1, 0, 2])  
                   for output_and_state in outputs_and_states]
        # Last step
        # n_labels x ( batch_size, size )
        outputs = [tf.gather(output, int(output.get_shape()[0]) - 1) 
                   for output in outputs]
        
        # n_labels x ( batch_size, cell.state_size )
        self._final_state = [output_and_state[1]\
                   for output_and_state in outputs_and_states]
        
        # self.n_labels x ( batch_size, n_classes )
        self.logits = logits = {}
        
        for slot in self.node_types:
            n_classes = len(self.dictionaries[slot])
            with tf.variable_scope("output_" + slot):
                #if hidden_layer:
                weight = tf.get_variable("weight", [size, n_classes])
                #else:
                #    weight = tf.get_variable("weight", [n_input, n_classes])
                bias = tf.get_variable("bias", [n_classes])

                # ( batch_size, n_classes )
                logit = tf.matmul(outputs[i], weight) + bias
            
            # logits
            logits[slot] = logit

        

        if self.tree != None:
            log_sum = self.tree.sum_over(crf_weight, logits)
        else:
            log_sum = self.sum_over()
        
        if has_output:
            if self.tree != None:
                logit_correct = self.tree.calculate_logit_correct(crf_weight, batch_size, logits, self._targets)
            else:
                logit_correct = self.calculate_logit_correct()
            
            # self.n_labels x (batch_size)
            loss_coefficients = []
            for id, slot in enumerate(self.node_types):
                # (batch_size)
                loss_coefficients.append( tf.gather( self.loss_weights[slot], self._targets[:,id] ))

            # (self.n_labels, batch_size)
            loss_coefficients = tf.stack ( loss_coefficients )

            self._cost =  tf.reduce_mean(tf.reduce_sum( (log_sum - logit_correct) * loss_coefficients, 0))
            
        if is_training:
            self.make_train_op( )
        else:
            if has_output:
                self.make_test_op( )
            else:
                self.make_prediction_op()
    
        self._saver =  tf.train.Saver()
    
    def sum_over( self ):
        '''
        Sum over the exponential term

        Return:
        -------
        log_sum:            numpy array of size = (self.n_labels, batch_size)
        '''

        log_sum = []

        for slot in self.node_types:
            # ( batch_size , n_classes)
            logit = self.logits[slot]

            # ( n_classes, batch_size )
            logit = tf.transpose(logit)
                    
            # ( batch_size )
            l = tf.reduce_min(logit, 0)
            l += tf.log(tf.reduce_sum(tf.exp(logit - l), 0))

            log_sum.append( l )

        return tf.stack( log_sum )

    def calculate_logit_correct( self ):
        '''
        Calculate the correct logit

        Return:
        -------
        logit_correct:       numpy array of size = (self.n_labels, batch_size)
        '''

        logit_correct = []
        for id, slot in enumerate(self.node_types):
            logit = self.logits[slot]

            logit_correct.append( gather_2d(logit
                , tf.transpose(tf.stack([tf.range(self.batch_size), self._targets[:,id] ]))) )

        return tf.stack( logit_correct )

    def predict( self ) :
        '''
        Predict the best combination (they are all independent here)
        
        Return:
        -------
        out:            numpy array of size = (batch_size, len(self.node_types) )
        '''

        max_logits = []
        for slot in self.node_types:
            # ( batch_size, n_classes )
            discounted_logit = self.logits[slot]

            max_logit = tf.argmax(discounted_logit, 1)

            max_logits.append(max_logit)

        # batch_size, len(self.node_types)
        max_logits = tf.cast(tf.transpose(tf.stack(max_logits)), np.int32)

        return max_logits
        
    def make_train_op(self):
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        self._train_op = []
            
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          self.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        
    def make_test_op(self):
        # (batch_size, self.n_labels)
        if self.tree != None:
            out = self.tree.predict( self.crf_weight, self.batch_size, self.logits )
        else:
            out = self.predict()
        
        # (self.n_labels, batch_size)
        correct_preds = [tf.equal(out[:,i], self._targets[:,i]) \
                for i in xrange(self.n_labels)]

        # Return number of correct predictions as well as predictions
        self._test_op = (out, 
                         [tf.reduce_mean(tf.cast(correct_pred, np.float32)) \
                         for correct_pred in correct_preds])
        
    def make_prediction_op(self):
        # (batch_size, self.n_labels)
        if self.tree != None:
            out = self.tree.predict( self.crf_weight, self.batch_size, self.logits )
        else:
            out = self.predict()
        
        self._test_op = out
    
    
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))
    
    @property
    def debug(self):
        return self._debug
        
    @property
    def saver(self):
        return self._saver
    
    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op
    
    @property
    def test_op(self):
        return self._test_op
    
    @property
    def input_lengths(self):
        return self._input_lengths