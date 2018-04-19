'''
Created on Mar 12, 2017

@author: Tuan
'''
import numpy as np
import tensorflow as tf

class SentenceRecover(object):
    '''
    classdocs
    '''
    

    def __init__(self, is_training, has_output, config):
        '''
        Constructor
        '''
    
    
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