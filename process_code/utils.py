'''
Created on Mar 3, 2017

@author: Tuan
'''
import os


# default mode is to train and test at the same time
TRAIN = 'TRAIN'
VALIDATE = 'VALIDATE'
TEST = 'TEST'
BLIND_TEST = 'BLIND_TEST'

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.join(ROOT_DIR, "process_code")
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

FEATURE_TRAIN_DIR = os.path.join(ROOT_DIR, 'feature_vectors_reduced_train')
FEATURE_TEST_DIR = os.path.join(ROOT_DIR, 'feature_vectors_reduced_test')
FEATURE_VAL_DIR = os.path.join(ROOT_DIR, 'feature_vectors_reduced_val')
FEATURE_BLINDTEST_DIR = os.path.join(ROOT_DIR, 'feature_vectors_reduced_blindtest')

LSMDC_DATA = os.path.join(ROOT_DIR, 'LSMDC task')

ORIGINAL_ANNOTATE_TRAIN_FILE = os.path.join(LSMDC_DATA, 'LSMDC16_annos_training.csv')
ORIGINAL_ANNOTATE_VAL_FILE = os.path.join(LSMDC_DATA, 'LSMDC16_annos_val.csv')
ORIGINAL_ANNOTATE_TEST_FILE = os.path.join(LSMDC_DATA, 'LSMDC16_annos_test.csv')
ORIGINAL_BLINDTEST_FILE = os.path.join(LSMDC_DATA, 'LSMDC16_annos_blindtest.csv')

TUPLE_TRAIN_FILE = os.path.join(LSMDC_DATA, 'LSMDC16_annos_training.desc.lemma.txt')
TUPLE_VAL_FILE = os.path.join(LSMDC_DATA, 'LSMDC16_annos_val.desc.lemma.txt')
TUPLE_TEST_FILE = os.path.join(LSMDC_DATA, 'LSMDC16_annos_test.desc.lemma.txt')

START = 'start'
SUBJECT = 'subject'
VERB = 'verb'
OBJECT = 'object'
PREP = 'prep'
PREP_DEP = 'prepDep'
ALL_SLOTS = [SUBJECT, VERB, OBJECT, PREP, PREP_DEP]