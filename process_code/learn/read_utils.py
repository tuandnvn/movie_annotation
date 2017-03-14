'''
Created on Mar 5, 2017

@author: Tuan
'''
import glob
import os
import re

from collections import defaultdict
from gensim.corpora.dictionary import Dictionary
import numpy as np
from utils import ALL_SLOTS


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
    for clip_file, feature_file_name in feature_clip_and_files:
        try:
            array = np.load(feature_file_name)
            
            yield (clip_file, array)
        except:
            print 'Load file ', clip_file, ' has problem!!!'



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
    
    p = re.compile('(?P<sentence>\d+)(\t\s*)?(?P<subject>[\s\w]+)(,\s*)?(?P<verb>[\s\w]+)(,\s*)?(?P<object>[\s\w]+)(,\s*)?(?P<prep>[\s\w]+)?(,\s*)?(?P<prepDep>[\s\w]+)?')
    
    documents = {}
    for t in ALL_SLOTS:
        documents[t] = []
        
    with open(tuple_file, 'r') as fh:
        for line in fh:
            result = p.match( line.strip() )
            
            for t in ALL_SLOTS:
                documents[t].append(result.group(t).lower() if result.group(t) != None else 'null')
    
    for t in ALL_SLOTS:
        ds[t].add_documents([documents[t]])
                
        ds[t].filter_extremes(no_below=20)
        ds[t].compactify()
    
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
    p = re.compile('(?P<sentence>\d+)(\t\s*)?(?P<subject>[\s\w]+)(,\s*)?(?P<verb>[\s\w]+)(,\s*)?(?P<object>[\s\w]+)(,\s*)?(?P<prep>[\s\w]+)?(,\s*)?(?P<prepDep>[\s\w]+)?')
    
    with open(tuple_file, 'r') as fh:
        for line in fh:
            result = p.match( line.strip() )
            
            sentence = int(result.group('sentence'))
        
            values = dict([ (t, result.group(t).lower()) if result.group(t) != None else 'null' for t in ALL_SLOTS ])
            
            # Turn token in values to ids
            # If token is not found, turn its to null's id
            ids = [ds[t][values[t]] if values[t] in ds[t] else ds[t]['null'] for t in ALL_SLOTS]
             
            if sentence in clip_name_map:
                labels[clip_name_map[sentence]].append(ids)
            
    return labels