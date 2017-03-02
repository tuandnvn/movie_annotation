'''
Created on Feb 27, 2017

@author: Tuan
'''
'''

'''

import glob
import os
import pickle

import argparse
from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A script to process the features from CGG network into lower dimensions')
    
    parser.add_argument('-p', '--process', nargs = 6, metavar = ('INPUT_DIR', 'MODEL', 'CLIP_NAME_FILE', 'ALGORITHM', 'COMPONENTS', 'LIMIT'),
                                help = "Process INPUT_DIR that stores the feature vectors. One feature vector file corresponds to one clip. \
                                Generate a decomposition model, save it down to MODEL file. Only use clips in CLIP_NAME_FILE. Limit for the first n clips." )
    
    parser.add_argument('-g', '--generate',  nargs = 5, metavar = ('OUTPUT_DIR', 'MODEL', 'CLIP_NAME_FILE', 'ALGORITHM', 'LIMIT'),
                                help = "OUTPUT_DIR that stores the feature vectors (in reduced dimensions).\
                                Loading ALGORITHM model from MODEL file." )
    
    parser.add_argument('-v', '--verbose',  action='store_true', metavar = ('VERBOSE'),
                                help = "Whether to print out information." )
    
    args = parser.parse_args()
    
    verbose = args.verbose
    
    if args.process and len(args.process) == 6:
        input_directory = args.process[0]
        model_file = args.process[1]
        clip_name_file = args.process[2]
        algorithm = args.process[3]
        components = args.process[4]
        limit = args.process[5]
        
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
        
        trainer = None
        if algorithm == 'IPCA':
            trainer = IncrementalPCA(n_components=components)
        
        if trainer != None:
            counter = 0
            
            for clip_name in clip_names[:limit]:
                feature_file = feature_file_map[clip_name]
                array = np.load(feature_file)
                
                trainer.partial_fit(array)
                counter += 1
                
                if counter % 100 == 0:
                    print counter
                
                if verbose:
                    print clip_name
            
            print 'Dump model to ', model_file
            # Save down model into model_file
            joblib.dump(trainer, model_file)
    
    if args.generate and len(args.process) == 5:
        output_directory = args.process[0]
        model_file = args.process[1]
        clip_name_file = args.process[2]
        algorithm = args.process[3]
        limit = args.process[4]
        
        clip_names = []
        with open(clip_name_file, 'r') as file_handler:
            for line in file_handler:
                clip_name, _ = line.split()
                clip_names.append(clip_name)
        
        trainer = joblib.load(model_file)
        
        if trainer != None:
            for clip_name in clip_names[:limit]:
                feature_file = feature_file_map[clip_name]
                array = np.load(feature_file)
                
                reduced_array = trainer.transform(array)
                new_feature_file = os.path.join(output_directory, clip_name + '.rfv')
                
                if verbose:
                    print new_feature_file, reduced_array.shape
                np.save(new_feature_file, reduced_array)
    
    
    
    
    
        
    
    
    