'''
Created on Mar 4, 2017

@author: Tuan
'''
from learn.crf_tree import CRFTree
from utils import SUBJECT, OBJECT, PREP, ALL_SLOTS, VERB, PREP_DEP, START


class NoTreeConfig(object):
    init_scale = 0.1
    learning_rate = 0.5     # Set this value higher without norm clipping
                            # might make the cost explodes
    max_grad_norm = 5       # The maximum permissible norm of the gradient
    num_layers = 1          # Number of LSTM layers
    num_steps = 20          # Divide the data into num_steps segment 
    hidden_size = 200       # the number of LSTM units
    max_epoch = 10          # The number of epochs trained with the initial learning rate
    max_max_epoch = 500     # Number of running epochs
    keep_prob = 0.8         # Drop out keep probability, = 1.0 no dropout
    lr_decay = 0.980         # Learning rate decay
    batch_size = 100         # We could actually still use batch_size for convenient
    hop_step = 5            # Hopping between two samples
    test_epoch = 10         # Test after these many epochs
    save_epoch = 10
    n_input = 500
        
class TreeConfig(NoTreeConfig):
    crf_weight = 1
    
    def __init__(self, gensim_dictionaries):
        NoTreeConfig.__init__(self)
        
        # Create tree 
        dictionaries = {}
        for key in ALL_SLOTS:
            dictionaries[key] = gensim_dictionaries[key].word2id
        
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
        
        self.tree = CRFTree( ALL_SLOTS, dictionaries, edges )
        
class TreeWithStartConfig(NoTreeConfig):
    crf_weight = 1
    
    def __init__(self, gensim_dictionaries):
        
        NoTreeConfig.__init__(self)
        
        # Create tree 
        dictionaries = {}
        for key in ALL_SLOTS:
            dictionaries[key] = gensim_dictionaries[key].word2id
        dictionaries[START] = { 'start' : 0 }
        
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
        
        self.tree = CRFTree( [START] + ALL_SLOTS, dictionaries, edges )