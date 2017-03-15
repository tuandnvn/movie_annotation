'''
Created on Mar 5, 2017

@author: Tuan
'''
import glob
import os
import re

from collections import defaultdict, Counter
import numpy as np
from utils import ALL_SLOTS, PREP_DEP

class Dictionary(object):
    def __init__(self):
        self.counter = Counter()
        self.token2id = {}

    def add_document(self, document):
        '''
        Parameters:
        document: List of string
        '''
        self.counter.update(set(document))
        for word in document:
            if word not in self.token2id:
                self.token2id[word] = len(self.token2id)

    def add_documents(self, documents):
        for document in documents:
            self.add_document(document)

    def filter_extremes(self, no_below=10, no_above=1):
        original_length = len(self.counter)

        tokens = self.counter.keys()

        total_count = sum (self.counter.values())

        for token in tokens:
            if self.counter[token] < no_below or self.counter[token] > no_above *  total_count:
                 del self.counter[token]
                 del self.token2id[token]

    def compactify( self ):
        for id, token in enumerate(sorted(self.token2id)):
            self.token2id[token] = id

    def __str__ (self):
        stuffs = []
        for token in sorted(self.token2id):
            stuffs.append(token + ' : ' + str(self.token2id[token]))
        return ', '.join(stuffs) 

    def __len__(self):
        return len(self.token2id)

    @property
    def id2token(self):
        if '_id2token' not in self.__dict__:
            self._id2token = dict([ (value, key) for key, value in self.token2id.iteritems() ])
        return self._id2token
    
def ids_to_values ( ids, ds ):
    '''
    Parameters:
    ----------
    ids:    list of ids
    ds:     My Dictionaries that store words in the tuples (use create_vocabulary to create them)
    
    Return:
    -------
    values: list of string
    '''
    values = [ds[t].id2token[ids[i]] if ids[i] in ds[t].id2token else 'ERROR' for i, t in enumerate(ALL_SLOTS)]

    return values

def values_to_ids ( values , ds):
    '''
    Parameters:
    ----------
    values: Dictionary from slot to string
    ds:     My Dictionaries that store words in the tuples (use create_vocabulary to create them)
    
    Return:
    -------
    ids:    list of ids
    '''
    ids = [ds[t].token2id[values[t]] if values[t] in ds[t].token2id else ds[t].token2id['null'] for t in ALL_SLOTS]

    return ids


def read_feature_vectors( directory ):
    '''
    Generate clip_file name and feature vector 
    
    Parameters:
    ----------
    directory: Where you store the feature vectors
    
    Yield:
    ------
    clip_file: file name without extension
    array:     numpy array of shape (#frame x feature_size)
    '''
    feature_clip_and_files = []
    extension = '.rfv.npy'
    feature_files = glob.glob(os.path.join(directory, '*' + extension))
    for feature_file in feature_files:
        ff = feature_file.split("/")[-1]
        feature_clip_and_files.append( (ff[:- len(extension)], feature_file) )
    
    # Clip feature should be generated according to the alphanumerical order
    counter = 0

    for clip_file, feature_file_name in feature_clip_and_files:
        try:
            array = np.load(feature_file_name)
            
            yield (clip_file, array)
            counter += 1
        except:
            print 'Load file ', clip_file, ' has problem!!!'

        if counter > 5000:
            return

def extract( line ):
    p = re.compile('(?P<sentence>\d+)(\s*)?(?P<subject>\w+[\s\w]?)(,\s*)?(?P<verb>[\s\w]+)(,\s*)?(?P<object>[\s\w]+)(,\s*)?(?P<prep>[\s\w]+)?(,\s*)?(?P<prepDep>[\s\w]+)?')

    result = p.match( line.strip() )

    return result

def create_vocabulary( tuple_file ):
    '''
    Create a mapping between vocabulary and ids in tuple_file
    Should only keep terms that have term frequency >= 20
    
    Parameters:
    ----------
    tuple_file:  The file that stores tuples
                    format =   1       someone,      kick,      null,      over,  trashcan
    
    Return:
    ------
    dictionaries: gensim Dictionaries that store words in the tuples
    '''
    ds = {}
    for t in ALL_SLOTS:
        ds[t] = Dictionary()
    
    documents = {}
    for t in ALL_SLOTS:
        documents[t] = []
        
    with open(tuple_file, 'r') as fh:
        for line in fh:
            result = extract(line)
            
            for t in ALL_SLOTS:
                u = [result.group(t).lower() if result.group(t) != None else 'null']
                documents[t].append(u)

    
    for t in ALL_SLOTS:
        ds[t].add_documents(documents[t])

        print ds[t].counter['null']
        
        ds[t].filter_extremes(no_below=20, no_above=1)
        ds[t].compactify()

        print len(ds[t])
        
    
    return ds
            
def read_original_file( original_file ):
    '''
    Parameters:
    ----------
    original_file:    format = 3001_21_JUMP_STREET_00.02.55.644-00.02.56.718    00.02.55.644    00.02.56.718    00.02.55.181    00.02.57.181
    
    Return:
    ------
    clip_name_map:    dictionary from order of clip in file to clip name file (without extension)
    '''
    clip_name_map = {}
    
    sentence = 1
    with open(original_file, 'r') as fh:
        for line in fh:
            clip_name = line.strip().split('\t') [0]
            clip_name_map[sentence] = clip_name
            sentence += 1
    
    return clip_name_map

def read_output_labels ( original_file, tuple_file, ds ):
    '''
    Parameters:
    ----------
    original_file:    format = 3001_21_JUMP_STREET_00.02.55.644-00.02.56.718    00.02.55.644    00.02.56.718    00.02.55.181    00.02.57.181    Now outside, SOMEONE kicks over a trashcan.
    tuple_file:        format =   1       someone,      kick,      null,      over,  trashcan
    ds:                gensim Dictionaries that store words in the tuples (use create_vocabulary to create them)
    
    Return:
    ------
    labels: Dictionary from clip_name (no extension) to label
    '''
    clip_name_map = read_original_file( original_file )
    
    labels = defaultdict(list)

    #for t in ds:
    #    print t
    #    print ds[t]

    with open(tuple_file, 'r') as fh:
        for line in fh:
            result = extract(line)
            
            sentence = int(result.group('sentence'))
        
            values = dict([ (t, result.group(t).lower()) if result.group(t) != None else 'null' for t in ALL_SLOTS ])

            #sprint values
            
            # Turn token in values to ids
            # If token is not found, turn its to null's id
            ids = values_to_ids( values, ds )
            
            #print ids
            if sentence in clip_name_map:
                labels[clip_name_map[sentence]].append(ids)
            
    return labels