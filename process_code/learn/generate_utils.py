'''
Created on Mar 3, 2017

@author: Tuan
'''
import numpy as np


def gothrough(feature_generator, labels, batch_size, max_steps):
    """
    Interate through 
    Parameters:
    ----------
    feature_generator: generator, see read_utils.read_feature_vectors
    labels:            labels: Dictionary from clip_name (no extension) to label
    batch_size:
    max_steps:
    
    Yields:
    -------
    
    x: features in the shape of [batch_size, max_steps, data_point_size] (with zero-padding at the end)
    y: output in the shape of [batch_size, num_labels]
    z: length of each sequence in the shape of [batch_size]
    """
    data_point_size = None
    num_labels = None
    
    features = []
    generated_lbls = []
    lengths = []
    
    for (clip_file, feature_array) in feature_generator:
        no_of_frames, data_size = feature_array.shape
        
        # There are multiple label set for each clip_file
        lbls = labels[clip_file]
        
        if data_point_size == None:
            data_point_size = data_size
        
        for lbl in lbls:
            if num_labels == None:
                num_labels = len(lbl)
            features.append(feature_array)
            generated_lbls.append(lbl)
            lengths.append(no_of_frames)
        
            if len(features) == batch_size:
                x = np.zeros( (batch_size, max_steps, data_point_size), dtype=np.float32 )
                y = np.zeros( (batch_size, num_labels), dtype=np.int32 )
                z = np.array( lengths, dtype = np.int32 )
                
                yield x, y, z
        
        
        