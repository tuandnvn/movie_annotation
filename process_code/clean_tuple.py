'''
Created on Mar 1, 2017

@author: Tuan
'''
'''
Using a spacy parser, turn a tuple form into a lemmatized form
Example is:
( swarm,  continues,  null, on,  buildings )
-> ( swarm,  continue,  null, on,  buildings ) 
Note that buildings should be kept intact because 
buildings appear a lots in the training dataset -> other words might be similar
are "windows", "doors", "people"

We need to feed in a counter of words because 
we want to keep the non-lemmatized form of words if its appears a lots)
- For example: 
        + plural forms of common words 
        + adverbs (sharply, directly etc.)
- These are considered as different words, so hopefully that could improve the performance
Parameters
----------
spacy_parser: a spacy parser, in this case should be an english parser
counter: counter of terms in the training data 
                       
Returns
-------
'''

from collections import Counter
import re

import numpy as np
import argparse

from spacy.en import English

def process ( spacy_parser, c, subject, verb, obj, prep, prepDep ):
    result = ["null"] * 5
    
    try:
        ts =  [t.lower() if not t.isspace() else 'null' for t in [subject, verb, obj, prep, prepDep]]
        
        simple_sentence = unicode('%s %s %s %s %s' % tuple(ts))
        
        # For each word, count the tokens
        no_tokens = [len(t.split()) for t in ts]
        
        lemmatized_words = []
        
        # Token counter
        token_counter = 0
        
        # Word counter
        word_counter = 0
        
        lemma = []
        for token in spacy_parser(simple_sentence):
            token_counter += 1
            lemma.append(token.lemma_)
            
            if token_counter == no_tokens[word_counter]:
                lemmatized_words.append(' '.join(lemma))
                
                # Reset all variables
                word_counter += 1
                token_counter = 0
                lemma = []
        
        verb = lemmatized_words[1]
        
        if verb[:3] == 'be ':
            verb = verb[3:]
            
        result[1] = verb
        for i in [0, 2, 3, 4]:
            # The count of this form is significant in compare to the lemmatized form
            # Such as buildings vs building
            if counter[lemmatized_words[i]] == 0 or \
                (c[ts[i]] > 10 and float(c[ts[i]])/ float(c[lemmatized_words[i]]) > 0.2) or \
                (c[ts[i]] > 50 and float(c[ts[i]])/ float(c[lemmatized_words[i]]) > 0.1):
                result[i] = str(ts[i])
            else:
                result[i] = str(lemmatized_words[i])
    except Exception as e:
        print e
        print 'Problem in sentence ' + simple_sentence
        for token in spacy_parser(simple_sentence):
            print token.lemma_
            
    return result

'''
tuples = [(sentence, subject, verb, obj, prep, prepDep) ] 
'''
def printStatistics( tuples ):
    counter = Counter()
    slot_counters = [Counter() for _ in xrange(5)]
    
    
    for t in tuples:
        counter.update(t)
        
        for i in xrange(5):
            # Start from 1
            slot_counters[i].update((t[i + 1], ))
        
    print 'Number of unique words = ', len(counter)
    for value in [1, 5, 10, 20, 50, 100, 200, 500, 1000]:
        l = [t for t in counter if counter[t] > value]
        print 'Number of words with frequency > ', value, ' = ', len(l)
        if value == 100:
            print sorted((counter[u], u) for u in l)
            
    for i, t in enumerate(['subject', 'verb', 'object', 'prep', 'prepDep']):
        print '----------------------------------------------------------------\n'
        
        print 'Number of unique words for ', t, ' = ', len(slot_counters[i])
        for value in [1, 5, 10, 20, 50, 100, 200, 500, 1000]:
            l = [t for t in slot_counters[i] if slot_counters[i][t] > value]
            print 'Number of words with frequency > ', value, ' = ', len(l)
            if value == 100:
                print sorted((slot_counters[i][u], u) for u in l)
                
    return counter, slot_counters
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A script to clean tuple forms to either simpler form(lemmatize) or to remove rare words')
    
    parser.add_argument('-i', '--input_file', action='store', metavar = ('INPUT_FILE'),
                                help = "Input file that store tuples." )
    
    parser.add_argument('-o', '--output_file', action='store', metavar = ('OUTPUT_FILE'),
                                help = "Output file that store tuples." )
    
    parser.add_argument('-v', '--verbose',  action='store_true',
                                help = "Whether to print out information." )
    
    parser.add_argument('-l', '--limit',  action='store', metavar = ('LIMIT'),
                                help = "Limit to the first LIMIT lines." )
    
    args = parser.parse_args()
    
    input_file = args.input_file
    output_file = args.output_file
    verbose = args.verbose
    limit = int(args.limit)
    
    p = re.compile('(?P<sentence>\d+)(\t\s*)?(?P<subject>[\s\w]+)(,\s*)?(?P<verb>[\s\w]+)(,\s*)?(?P<object>[\s\w]+)(,\s*)?(?P<prep>[\s\w]+)?(,\s*)?(?P<prepDep>[\s\w]+)?')
    
    outputs = []
    
    
    
    lines = []
    '''First pass, just do some statistics'''
    with open(input_file, 'r') as filehandler:
        for line in filehandler:
            lines.append(line.strip())
    
    parse_tuples = []
    
    for line in lines:
        result = p.match( line )
        sentence = result.group('sentence')
        
        values = [ result.group(t).lower() if result.group(t) != None else 'null' for t in ['subject', 'verb', 'object', 'prep', 'prepDep'] ]
        parse_tuples.append( tuple([sentence] + values ))
    
    
    counter, slot_counters = printStatistics(parse_tuples)
        
    parser = English()
    
    processed_tuples = []
    
    '''Second pass, do the processing'''
    for sentence, subject, verb, obj, prep, prepDep  in parse_tuples[:limit]:
        values = process ( parser, counter, subject, verb, obj, prep, prepDep )
        
        if verbose:
            print sentence, values
        if any(values):
            processed_tuples.append( tuple([sentence] + values ))
            
            
    printStatistics(processed_tuples)
                
    with open(output_file, 'w') as filehandler:
        for t in processed_tuples:
            filehandler.write('%s\t%10s,%10s,%10s,%10s,%10s' % t)
            filehandler.write('\n')