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

def process ( spacy_parser, counter, subject, verb, obj, prep, prepDep ):
    tuple =  [subject, verb, obj, prep, prepDep]
    subject, verb, obj, prep, prepDep = subject.lower(), verb.lower(), obj.lower(), prep.lower(), prepDep.lower()
    
    simple_sentence = '%s %s %s %s %s' % (subject, verb, obj, prep, prepDep)
    simple_sentence = unicode(simple_sentence.lower())
    
    # For each word, count the tokens
    no_tokens = [len(t.split()) for t in [subject, verb, obj, prep, prepDep]]
    
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
    
    result = (None, None, None, None, None)
    result[1] = lemmatized_words[1]
    for i in [0, 2, 3, 4]:
        # The count of this form is significant in compare to the lemmatized form
        # Such as buildings vs building
        if (counter[tuple[i]] > 10 and float(counter[tuple[i]])/ float(counter[lemmatized_words[i]]) > 0.2) or \
            (counter[tuple[i]] > 50 and float(counter[tuple[i]])/ float(counter[lemmatized_words[i]]) > 0.1):
            result[i] = tuple[i]
        else:
            result[i] = lemmatized_words[i]
            
    return result
    
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
    
    
    counter = Counter()
    '''First pass, just do some statistics'''
    with open(input_file, 'r') as filehandler:
        for line in input_file:
            result = p.match( line )
            sentence = result.group('sentence')
            
            counter.update([result.group('subject').lower(), result.group('verb').lower(), 
                            result.group('object').lower(), result.group('prep'.lower()), result.group('prepDep').lower()])
    
    hist = np.histogram(counter.values())
    print hist
    parser = English()
    
    '''Second pass, do the processing'''
    with open(input_file, 'r') as filehandler:
        for line in input_file:
            value = process ( parser, counter, result.group('subject'), result.group('verb'), result.group('object'), result.group('prep'), result.group('prepDep'))
            if any(value):
                outputs.append((sentence, value)) 
                
    with open(output_file, 'w') as filehandler:
        for sentence, value in outputs:
            filehandler.write(sentence)
            filehandler.write('%10s,%10s,%10s,%10s,%10s' % value)
            filehandler.write('\n')