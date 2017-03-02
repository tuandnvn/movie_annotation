'''
Created on Mar 1, 2017

@author: Tuan
'''
import argparse
import re

def process ( subject, verb, obj, prep, prepDep ):
    
    # By default, just remove this tuple
    return (None, None, None, None, None)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A script to clean tuple forms to either simpler form(lemmatize) or to remove rare words')
    
    parser.add_argument('-i', '--input_file', action='store', metavar = ('INPUT_FILE'),
                                help = "Input file that store tuples." )
    
    parser.add_argument('-o', '--output_file', action='store', metavar = ('OUTPUT_FILE'),
                                help = "Output file that store tuples." )
    
    args = parser.parse_args()
    
    input_file = args.input_file
    output_file = args.output_file
    
    p = re.compile('<?P<sentence>\d+)\s+(?P<subject>\w+,)(?P<verb>\w+,)(?P<object>\w+,)(?P<prep>\w+,)(?P<prepDep>\w+)')
    
    outputs = []
    with open(input_file, 'r') as filehandler:
        for line in input_file:
            result = p.match( line )
            sentence = result.group('sentence')
            
            value = process ( result.group('subject'), result.group('verb'), result.group('object'), result.group('prep'), result.group('prepDep'))
            if any(value):
                outputs.append((sentence, value)) 
                
    with open(output_file, 'w') as filehandler:
        for sentence, value in outputs:
            filehandler.write(sentence)
            filehandler.write('%10s,%10s,%10s,%10s,%10s' % value)
            filehandler.write('\n')
            