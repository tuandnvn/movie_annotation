'''
Created on Feb 27, 2017

@author: Tuan
'''
'''

'''

import glob
import os

import argparse
from sklearn.decomposition import IncrementalPCA

import numpy as np
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A script to process the features from CGG network into lower dimensions')

    parser.add_argument('-i', '--input_directory', action='store', metavar = ('INPUT_DIR'),
                                help = "The input directory that stores the feature vectors. One feature vector file correspond to one clip." )
    
    parser.add_argument('-o', '--output_directory', action='store', metavar = ('OUTPUT_DIR'),
                                help = "The output directory that stores the feature vectors (in reduced dimensions). One feature vector file correspond to one clip." )
    
    parser.add_argument('-f', '--clip_name_file', action='store', metavar = ('CLIP_NAME_FILE'),
                                help = "File that store clip names." )
    
    parser.add_argument('-a', '--algorithm', action='store', metavar = ('ALGORITHM'),
                                help = "Dimension reduction algorithms to choose from. The supported ones are Incremental PCA (IPCA)")
    
    
    args = parser.parse_args()
    
    clip_name_file = args.clip_name_file
    input_directory = args.input_directory
    output_directory = args.output_directory
    algorithm = args.algorithm
    
    
    feature_files = glob.glob(os.path.join(input_directory, '*.fv.npy') )
    feature_file_map = {}
    
    for feature_file in feature_files:
        ff = feature_file.split("/")[-1]
        feature_file_map[ff[:- len('.fv.npy')]] = feature_file
    
    clip_names = []
    with open(clip_name_file, 'r') as file_handler:
        for line in file_handler:
            clip_name, _ = line.split()
            clip_names.append(clip_name)
    
    pca = IncrementalPCA(n_components=500)
    
    counter = 0
    
    size = 1000
    for clip_name in clip_names[:1000]:
        feature_file = feature_file_map[clip_name]
        array = np.load(feature_file)
        
        pca.partial_fit(array)
        counter += 1
        
        if counter % 100 == 0:
            print counter
    
    for clip_name in clip_names[:1000]:
        feature_file = feature_file_map[clip_name]
        array = np.load(feature_file)
        
        reduced_array = pca.transform(array)
        new_feature_file = os.path.join(output_directory, clip_name + '.rfv')
        
        np.save(new_feature_file, reduced_array)
    
    
    
    
    
        
    
    
    