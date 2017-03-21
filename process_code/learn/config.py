'''
Created on Mar 4, 2017

@author: Tuan
'''
from crf_tree import CRFTree
from utils import SUBJECT, OBJECT, PREP, ALL_SLOTS, VERB, PREP_DEP, START
import numpy as np
import tensorflow as tf

class NoTreeConfig(object):
    init_scale = 0.1
    learning_rate = 1     # Set this value higher without norm clipping
                            # might make the cost explodes
    max_grad_norm = 1       # The maximum permissible norm of the gradient
    num_layers = 5          # Number of LSTM layers
    num_steps = 20          # Divide the data into num_steps segment 
    hidden_size = 512       # the number of LSTM units
    max_epoch = 20          # The number of epochs trained with the initial learning rate
    max_max_epoch = 1000     # Number of running epochs
    keep_prob = 1.0         # Drop out keep probability, = 1.0 no dropout
    lr_decay = 0.988         # Learning rate decay
    batch_size = 500         # We could actually still use batch_size for convenient
    hop_step = 5            # Hopping between two samples
    test_epoch = 20         # Test after these many epochs
    save_epoch = 100
    n_input = 500
    hidden_layer = True
    crf_weight = 1
    balance = True
    train_algo = tf.train.AdagradOptimizer
    # train_algo = tf.train.GradientDescentOptimizer

    # class Weight(object):
    #   # correct_not_null = 1      # Totally correct for a not-null word
    #   # correct_null_null = 0.2   # Totally correct from null to null (should not be high, otherwise logit would stuck)
    #   # mistake_to_null = 0       # No point from correct to null  
    #   # mistake_not_null = 0.05   # It is better to predict incorrectly then to produce null
    #   # mistake_from_null = 0.2   # We might even want to boost prediction from null to something else to cover 
    #                             # for the case we miss the slot because of parse mistake or filter the dictionary
    #   null = 0.1 # Much prefer output that differ from null

    # This is correspond to a scale of 1/e^2 for a combination
    # null_weights = 0.1
    
    def __init__(self, gensim_dictionaries, limited_dictionaries = {}):
        self.node_types = ALL_SLOTS
        self.dictionaries = gensim_dictionaries

        self.loss_weights = {}

        for node_type in self.node_types:
          dic = limited_dictionaries[node_type]
          self.loss_weights[node_type] = np.zeros( len(gensim_dictionaries[node_type]), dtype=np.float32 )

          for id in dic:
            self.loss_weights[node_type][id] = 1.0 / ( 1 + np.log(dic[id] ))

        
class TreeConfig(NoTreeConfig):
    def __init__(self, gensim_dictionaries, limited_dictionaries):
        NoTreeConfig.__init__(self, gensim_dictionaries, limited_dictionaries)
        
        
        '''
        SUJBECT ------  VERB  -------  OBJECT 
                         |
                         |
                         |
                        PREP --------- PREP_DEP
        '''
        
        edges = { SUBJECT: [VERB],
                  VERB : [SUBJECT, OBJECT, PREP],
                  OBJECT:  [VERB],
                  PREP: [VERB, PREP_DEP],
                  PREP_DEP: [PREP]  }
        
        d = {}
        for key in self.node_types:
            d[key] = self.dictionaries[key].token2id

        self.tree = CRFTree( self.node_types, d, edges )
        
class TreeWithStartConfig(NoTreeConfig):
    def __init__(self, gensim_dictionaries):
        
        NoTreeConfig.__init__(self, gensim_dictionaries)
        
        self.node_types.append(START)
        # Create tree 
        self.dictionaries[START] = { 'start' : 0 }
        
        '''
                       START
                         |
                         |
                         |
        SUJBECT ------  VERB  -------  OBJECT 
                         |
                         |
                         |
                        PREP --------- PREP_DEP
        '''
        
        edges = { START : [VERB],
                  SUBJECT: [VERB],
                  VERB : [START, SUBJECT, OBJECT, PREP],
                  OBJECT:  [VERB],
                  PREP: [VERB, PREP_DEP],
                  PREP_DEP: [PREP]  }

        d = {}
        for key in self.node_types:
            d[key] = self.dictionaries[key].token2id
        
        self.tree = CRFTree( self.node_types, d, edges )